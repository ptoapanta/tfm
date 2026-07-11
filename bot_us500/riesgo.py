# =============================================================
# riesgo.py — Gestion de riesgo del agente US500
#
# Traduce el motor de simulacion del notebook (funcion simular,
# celda "Gestion de riesgo") a un gestor con estado apto para
# operar en tiempo real. Reproduce las mismas reglas de la cuenta
# de evaluacion GetLeveraged Turbo Trade:
#
#   1. Objetivo de ganancia alcanzado        -> detener el agente
#   2. Drawdown trailing excedido            -> detener el agente
#   3. Filtro de volatilidad minima (ATR)    -> descartar señal
#   4. Stop buffer diario                    -> descartar señal
#   5. Regla de consistencia (20%)           -> descartar señal
#   6. Filtro del modelo ML                  -> descartar señal
#
# NOTA SOBRE EL DRAWDOWN: al igual que en el notebook, el piso del
# drawdown trailing se calcula sobre el BALANCE REALIZADO (equity_max
# de operaciones cerradas). GetLeveraged evalua ademas el equity
# flotante intradia; antes de operar en real conviene revisar si el
# broker exige tambien controlar el drawdown sobre equity abierto.
# =============================================================

import MetaTrader5 as mt5
import logging
from config import cfg

log = logging.getLogger("riesgo")


# =============================================================
# Verificaciones del entorno MT5 (previas a cualquier decision)
# =============================================================
def verificar_entorno() -> tuple[bool, str]:
    """
    Comprueba que MT5 este operativo y el mercado abierto.
    Estas condiciones son ajenas a la estrategia: garantizan que
    el broker puede recibir una orden.

    Retorna (True, "") si todo esta en condiciones, o
            (False, "motivo") en caso contrario.
    """
    if not mt5.terminal_info():
        return False, "MT5 no esta conectado"

    if mt5.account_info() is None:
        return False, "No se pudo obtener informacion de la cuenta"

    posiciones = mt5.positions_get(symbol=cfg.symbol)
    n_pos = len(posiciones) if posiciones else 0
    if n_pos >= cfg.max_posiciones:
        return False, (
            f"Posiciones abiertas ({n_pos}) >= "
            f"maximo permitido ({cfg.max_posiciones})"
        )

    tick = mt5.symbol_info_tick(cfg.symbol)
    if tick is None or tick.bid == 0 or tick.ask == 0:
        return False, f"Mercado cerrado o sin cotizacion para {cfg.symbol}"

    return True, ""


# =============================================================
# Gestor de riesgo con estado (equivalente a la funcion simular)
# =============================================================
class GestorRiesgo:
    """
    Mantiene el estado financiero que las reglas de GetLeveraged
    exigen seguir a lo largo de la jornada:

        balance     : capital realizado acumulado
        equity_max  : maximo historico del balance (piso del trailing)
        gan_total   : ganancia neta acumulada desde el inicio
        gan_dia     : ganancia del dia en curso
        per_dia     : perdida del dia en curso (valor positivo)
        mejor_dia   : mejor ganancia diaria registrada

    El agente debe llamar a:
        - nuevo_dia(fecha)          al cambiar de jornada
        - registrar_cierre(pnl, ..) cada vez que una posicion cierra
    """

    def __init__(self):
        self.balance    = cfg.balance_inicial
        self.equity_max = cfg.balance_inicial
        self.gan_total  = 0.0
        self.gan_dia    = 0.0
        self.per_dia    = 0.0
        self.mejor_dia  = 0.0
        self.dia_act    = None

    # ── Cambio de jornada ─────────────────────────────────────
    def nuevo_dia(self, fecha) -> None:
        """
        Cierra la contabilidad del dia anterior y reinicia los
        acumuladores diarios. Debe llamarse cuando cambia la fecha.
        """
        if self.dia_act is not None and self.gan_dia > self.mejor_dia:
            self.mejor_dia = self.gan_dia
        self.dia_act = fecha
        self.gan_dia = 0.0
        self.per_dia = 0.0

    # ── Registro de una operacion cerrada ─────────────────────
    def registrar_cierre(self, pnl: float, motivo: str) -> None:
        """
        Actualiza el estado tras cerrar una posicion.
        'motivo' es 'TP', 'SL' o 'TIEMPO', igual que en el notebook.
        """
        self.balance   += pnl
        self.gan_total += pnl

        if motivo == "TP":
            self.gan_dia += pnl
        elif motivo == "SL":
            self.per_dia += abs(pnl)
        else:  # cierre por tiempo: se imputa segun el signo del resultado
            if pnl >= 0:
                self.gan_dia += pnl
            else:
                self.per_dia += abs(pnl)

        if self.balance > self.equity_max:
            self.equity_max = self.balance

    # ── Condiciones globales que detienen al agente ───────────
    def estado_global(self) -> tuple[bool, str]:
        """
        Verifica las dos condiciones que terminan la evaluacion.
        Retorna (True, "") si el agente puede seguir operando, o
                (False, "motivo") si debe detenerse.
        """
        if (self.balance - cfg.balance_inicial) >= cfg.objetivo_usd:
            return False, (
                f"OBJETIVO ALCANZADO: +${self.balance - cfg.balance_inicial:,.0f} "
                f"(meta ${cfg.objetivo_usd:,.0f})"
            )

        piso = self.equity_max * (1 - cfg.dd_pct)
        if self.balance <= piso:
            return False, (
                f"DRAWDOWN EXCEDIDO: balance ${self.balance:,.0f} <= "
                f"piso ${piso:,.0f}"
            )

        return True, ""

    # ── Filtros previos a abrir una posicion ──────────────────
    def puede_abrir(self, atr: float, prob: float) -> tuple[bool, str]:
        """
        Aplica, EN EL MISMO ORDEN que el notebook, los filtros que
        deciden si una señal ya validada por la estrategia se puede
        ejecutar. Retorna (True, "") o (False, "motivo").
        """
        # Filtro 1: volatilidad minima. Con SL < sl_min_pts el spread
        # consume demasiado margen de la operacion.
        if atr <= 0 or atr * cfg.sl_atr_mult < cfg.sl_min_pts:
            return False, (
                f"ATR insuficiente: SL={atr * cfg.sl_atr_mult:.2f}pts "
                f"< minimo {cfg.sl_min_pts}pts"
            )

        # Filtro 2: stop buffer diario. El bot para al 50% del limite.
        if self.per_dia >= cfg.stop_buf_usd:
            return False, (
                f"Stop buffer diario alcanzado: "
                f"perdida del dia ${self.per_dia:,.0f} >= ${cfg.stop_buf_usd:,.0f}"
            )

        # Filtro 3: consistencia. Ningun dia debe concentrar mas del
        # 20% de la ganancia total acumulada.
        if self.gan_total > 0 and self.mejor_dia > 0:
            if self.gan_dia >= self.gan_total * cfg.consistencia_max:
                return False, (
                    f"Regla de consistencia: ganancia del dia "
                    f"${self.gan_dia:,.0f} >= {cfg.consistencia_max*100:.0f}% "
                    f"del total ${self.gan_total:,.0f}"
                )

        # Filtro 4: modelo de Machine Learning.
        if prob < cfg.umbral_buy:
            return False, (
                f"Probabilidad ML insuficiente: "
                f"{prob:.3f} < umbral {cfg.umbral_buy}"
            )

        return True, ""

    def resumen(self) -> dict:
        """Estado actual del gestor para el log del agente."""
        piso = self.equity_max * (1 - cfg.dd_pct)
        return {
            "balance":    round(self.balance, 2),
            "gan_total":  round(self.gan_total, 2),
            "gan_dia":    round(self.gan_dia, 2),
            "per_dia":    round(self.per_dia, 2),
            "equity_max": round(self.equity_max, 2),
            "piso_dd":    round(piso, 2),
            "margen_dd":  round(self.balance - piso, 2),
        }


# =============================================================
# Calculo de precios de entrada, SL y TP
# =============================================================
def calcular_sl_tp(precio_entrada: float, atr: float) -> tuple[float, float]:
    """
    Calcula SL y TP para una operacion LONG a partir del precio de
    entrada y el ATR de la vela de la señal, con los mismos multiplos
    usados en el etiquetado del notebook (SL = 0.5xATR, TP = 1.5xATR).

    Retorna (sl, tp) redondeados a los digitos del simbolo.
    """
    info    = mt5.symbol_info(cfg.symbol)
    digitos = info.digits if info is not None else 1

    sl = round(precio_entrada - cfg.sl_atr_mult * atr, digitos)
    tp = round(precio_entrada + cfg.tp_atr_mult * atr, digitos)
    return sl, tp


def resumen_cuenta() -> dict:
    """
    Resumen del estado real de la cuenta en el broker (para logging).
    Es independiente del estado interno del GestorRiesgo.
    """
    cuenta = mt5.account_info()
    if cuenta is None:
        return {}
    return {
        "balance": round(cuenta.balance, 2),
        "equity":  round(cuenta.equity, 2),
        "profit":  round(cuenta.profit, 2),
        "margen":  round(cuenta.margin, 2),
    }
