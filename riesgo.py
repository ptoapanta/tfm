# =============================================================
# riesgo.py — Gestión de riesgo del agente US500 v3
#
# Verifica CUATRO condiciones antes de permitir una operación.
# Si alguna falla, el agente espera al siguiente ciclo.
# =============================================================

import MetaTrader5 as mt5
import logging
from config import cfg

log = logging.getLogger("riesgo")


def puede_operar() -> tuple[bool, str]:
    """
    Verifica todas las condiciones de riesgo en orden.

    Retorna:
        (True,  "")       si puede abrir una nueva posición
        (False, "motivo") si no puede, con el motivo explicado
    """

    # ── Condición 1: MT5 conectado ────────────────────────────
    if not mt5.terminal_info():
        return False, "MT5 no está conectado"

    # ── Condición 2: Información de cuenta disponible ─────────
    cuenta = mt5.account_info()
    if cuenta is None:
        return False, "No se pudo obtener información de la cuenta"

    # ── Condición 3: Drawdown máximo no superado ──────────────
    if cuenta.balance > 0:
        drawdown = (cuenta.balance - cuenta.equity) / cuenta.balance
        if drawdown >= cfg.max_drawdown_pct:
            return False, (
                f"Drawdown actual {drawdown*100:.1f}% supera el límite "
                f"{cfg.max_drawdown_pct*100:.0f}% — agente pausado"
            )

    # ── Condición 4: No superar el máximo de posiciones ───────
    posiciones = mt5.positions_get(symbol=cfg.symbol)
    n_pos = len(posiciones) if posiciones else 0
    if n_pos >= cfg.max_posiciones:
        return False, (
            f"Posiciones abiertas ({n_pos}) >= "
            f"máximo permitido ({cfg.max_posiciones})"
        )

    # ── Condición 5: Mercado abierto ──────────────────────────
    tick = mt5.symbol_info_tick(cfg.symbol)
    if tick is None or tick.bid == 0 or tick.ask == 0:
        return False, f"Mercado cerrado o sin cotización para {cfg.symbol}"

    return True, ""


def calcular_precios_long() -> tuple[float, float, float]:
    """
    Calcula entry, SL y TP para una operación LONG.
    SL y TP basados en ATR — deben coincidir con el entrenamiento.

    Retorna:
        (entry, sl, tp)
    """
    tick = mt5.symbol_info_tick(cfg.symbol)
    if tick is None:
        log.error(f"No se pudo obtener tick para {cfg.symbol}")
        return 0.0, 0.0, 0.0

    # Intentar obtener info del símbolo
    info    = mt5.symbol_info(cfg.symbol)
    digitos = info.digits if info is not None else 2

    # Si el símbolo no está en Market Watch, agregarlo e intentar de nuevo
    if info is None:
        log.warning(f"{cfg.symbol} no está en Market Watch — agregando...")
        mt5.symbol_select(cfg.symbol, True)
        info    = mt5.symbol_info(cfg.symbol)
        digitos = info.digits if info is not None else 2

    # Obtener ATR de las últimas velas
    rates = mt5.copy_rates_from_pos(
        cfg.symbol, mt5.TIMEFRAME_M15, 0, cfg.atr_periodo + 5
    )

    if rates is None or len(rates) < cfg.atr_periodo:
        # Fallback: estimar ATR como spread * 10
        spread = tick.ask - tick.bid
        atr_v  = spread * 10 if spread > 0 else tick.ask * 0.001
        log.warning(f"Sin rates para ATR — usando estimación: {atr_v:.2f}")
    else:
        high  = rates["high"].astype(float)
        low   = rates["low"].astype(float)
        close = rates["close"].astype(float)
        tr    = []
        for i in range(1, len(rates)):
            tr.append(max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i]  - close[i - 1])
            ))
        atr_v = sum(tr[-cfg.atr_periodo:]) / cfg.atr_periodo

    entry = tick.ask
    sl    = round(entry - cfg.sl_atr_mult * atr_v, digitos)
    tp    = round(entry + cfg.tp_atr_mult * atr_v, digitos)

    return entry, sl, tp


def resumen_cuenta() -> dict:
    """
    Retorna un resumen del estado de la cuenta para logging.
    """
    cuenta = mt5.account_info()
    if cuenta is None:
        return {}

    drawdown = 0.0
    if cuenta.balance > 0:
        drawdown = (cuenta.balance - cuenta.equity) / cuenta.balance * 100

    return {
        "balance":   round(cuenta.balance, 2),
        "equity":    round(cuenta.equity, 2),
        "profit":    round(cuenta.profit, 2),
        "drawdown":  round(drawdown, 2),
        "margen":    round(cuenta.margin, 2),
    }
