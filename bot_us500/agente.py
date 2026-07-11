# =============================================================
# agente.py — Loop principal del agente de trading US500
#
# Integra los cuatro componentes del sistema:
#   features.py  -> genera la señal y las 6 features (rentabilidad)
#   ONNX         -> filtra la señal con el modelo Random Forest
#   riesgo.py    -> valida la señal contra las reglas de fondeo
#   ejecutor.py  -> envia la orden al broker
#
# COMO USAR:
#   1. Abrir MetaTrader 5 y conectarse al broker / prop firm.
#   2. Activar el trading algoritmico en MT5.
#   3. Colocar pipeline_completo.onnx junto a estos archivos.
#   4. Ejecutar:  python agente.py
#   5. Para detener:  Ctrl+C
# =============================================================

import asyncio
import logging
import sys
from datetime import datetime

import numpy as np
import MetaTrader5 as mt5
import onnxruntime as ort

from config import cfg
from features import obtener_rates, construir_features, evaluar_proximidad
from riesgo import (
    verificar_entorno, calcular_sl_tp, resumen_cuenta, GestorRiesgo
)
from ejecutor import enviar_orden_buy, cerrar_posicion


log = logging.getLogger("agente")


# ── Logging ───────────────────────────────────────────────────
def configurar_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-9s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(cfg.log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ── Inicializacion de MT5 ─────────────────────────────────────
def iniciar_mt5() -> bool:
    if not mt5.initialize():
        log.error(f"No se pudo conectar a MT5: {mt5.last_error()}")
        return False

    if not mt5.symbol_select(cfg.symbol, True):
        log.error(f"No se pudo seleccionar el simbolo {cfg.symbol}")
        return False

    cuenta   = mt5.account_info()
    terminal = mt5.terminal_info()

    log.info("=" * 60)
    log.info("AGENTE US500 — Donchian + CCI + Random Forest")
    log.info("=" * 60)
    log.info(f"Broker      : {cuenta.company}")
    log.info(f"Cuenta      : {cuenta.login} ({cuenta.currency})")
    log.info(f"Balance     : {cuenta.balance:.2f}")
    log.info(f"Terminal    : {terminal.name}")
    log.info(f"Simbolo     : {cfg.symbol} | TF: {cfg.timeframe}")
    log.info(f"Estrategia  : SOLO LONG | Donchian({cfg.don_p}) + "
             f"SMA({cfg.sma_p}) + CCI>{cfg.cci_umb:.0f}")
    log.info(f"SL/TP       : {cfg.sl_atr_mult}xATR / {cfg.tp_atr_mult}xATR")
    log.info(f"Lote        : {cfg.lote} | Umbral ML: {cfg.umbral_buy}")
    log.info(f"Cuenta eval : ${cfg.balance_inicial:,.0f} | "
             f"objetivo +${cfg.objetivo_usd:,.0f} | DD -{cfg.dd_pct*100:.0f}%")
    log.info("=" * 60)
    return True


# ── Carga del modelo ONNX ─────────────────────────────────────
def cargar_modelo() -> ort.InferenceSession:
    try:
        sesion = ort.InferenceSession(
            cfg.modelo_path,
            providers=["CPUExecutionProvider"]
        )
        entrada = sesion.get_inputs()[0]
        log.info(f"Modelo ONNX cargado: {cfg.modelo_path}")
        log.info(f"Input: {entrada.name} | Forma: {entrada.shape} "
                 f"(scaler incluido en el pipeline)")
        return sesion
    except FileNotFoundError:
        log.error(
            f"Modelo '{cfg.modelo_path}' no encontrado. "
            f"Copia el archivo .onnx a la carpeta del agente."
        )
        sys.exit(1)
    except Exception as e:
        log.error(f"Error al cargar el modelo ONNX: {e}")
        sys.exit(1)


# ── Inferencia ────────────────────────────────────────────────
def inferir(sesion: ort.InferenceSession, features: np.ndarray) -> float:
    """
    Ejecuta el pipeline ONNX (scaler + Random Forest) sobre las
    features CRUDAS y devuelve la probabilidad de la clase 1 (que la
    operacion alcance el TP antes que el SL).
    """
    nombre  = sesion.get_inputs()[0].name
    entrada = features.reshape(1, -1).astype(np.float32)
    salida  = sesion.run(None, {nombre: entrada})

    # La salida de probabilidad es una secuencia de mapas {clase: prob}
    # por el ZipMap del pipeline. Extraemos la probabilidad de clase 1.
    prob = salida[1][0]
    if isinstance(prob, dict):
        return float(prob[1])
    return float(prob[1])   # tambien valido si viniera como array


# ── Estado de la posicion abierta ─────────────────────────────
class Posicion:
    """Guarda los datos de la posicion actualmente abierta por el agente."""
    def __init__(self):
        self.ticket = None
        self.entrada = 0.0
        self.hora_entrada = None

    @property
    def abierta(self) -> bool:
        return self.ticket is not None

    def abrir(self, ticket, entrada, hora):
        self.ticket = ticket
        self.entrada = entrada
        self.hora_entrada = hora

    def cerrar(self):
        self.ticket = None
        self.entrada = 0.0
        self.hora_entrada = None


def _pnl_realizado(ticket: int) -> float:
    """
    Recupera el beneficio realizado de la posicion cerrada a partir
    del historico de operaciones de MT5.
    """
    desde = datetime(2000, 1, 1)
    hasta = datetime.now()
    deals = mt5.history_deals_get(desde, hasta, position=ticket)
    if not deals:
        return 0.0
    return float(sum(d.profit for d in deals))


# ── Gestion de la posicion abierta ────────────────────────────
def gestionar_posicion(pos: Posicion, gestor: GestorRiesgo, ahora) -> None:
    """
    Comprueba si la posicion sigue viva en el broker. Si el broker la
    cerro (TP o SL), contabiliza el resultado. Si supero el tiempo
    maximo, la cierra manualmente.
    """
    vivas = mt5.positions_get(ticket=pos.ticket)

    if not vivas:
        # El broker cerro la posicion (toco TP o SL).
        pnl    = _pnl_realizado(pos.ticket)
        motivo = "TP" if pnl > 0 else "SL"
        gestor.registrar_cierre(pnl, motivo)
        log.info(f"Posicion {pos.ticket} cerrada por {motivo} | "
                 f"PnL ${pnl:,.2f} | balance ${gestor.balance:,.2f}")
        pos.cerrar()
        return

    # Sigue abierta: comprobar el maximo de tiempo permitido.
    velas = (ahora - pos.hora_entrada).total_seconds() / (15 * 60)
    if velas >= cfg.tiempo_max_velas:
        log.info(f"Posicion {pos.ticket} supero {cfg.tiempo_max_velas} velas "
                 f"— cierre por tiempo")
        if cerrar_posicion(pos.ticket):
            pnl = _pnl_realizado(pos.ticket)
            gestor.registrar_cierre(pnl, "TIEMPO")
            log.info(f"Cierre por tiempo | PnL ${pnl:,.2f} | "
                     f"balance ${gestor.balance:,.2f}")
            pos.cerrar()


# ── Un ciclo completo ─────────────────────────────────────────
async def ciclo(sesion, gestor: GestorRiesgo, pos: Posicion, n: int) -> bool:
    """
    Ejecuta un ciclo del agente. Retorna False si el agente debe
    detenerse (objetivo alcanzado o drawdown excedido).
    """
    ahora = datetime.now()
    fecha = ahora.date()
    log.info(f"--- Ciclo #{n} | {ahora.strftime('%Y-%m-%d %H:%M:%S')} ---")

    # Cambio de jornada.
    if fecha != gestor.dia_act:
        gestor.nuevo_dia(fecha)
        log.info(f"Nueva jornada: {fecha}")

    # Condiciones globales que terminan la evaluacion.
    seguir, motivo = gestor.estado_global()
    if not seguir:
        log.info(f"FIN DE LA OPERATIVA: {motivo}")
        return False

    # Si hay una posicion abierta, gestionarla y no buscar nuevas señales.
    if pos.abierta:
        gestionar_posicion(pos, gestor, ahora)
        return True

    # Descargar datos.
    df = obtener_rates()
    if df is None:
        log.warning("Sin datos suficientes — se omite el ciclo")
        return True

    # Calcular features y detectar señal en la ultima vela cerrada.
    features, hay_senal, ctx = construir_features(df)

    if not hay_senal:
        prox = evaluar_proximidad(df)
        log.info(f"Sin señal | {prox['descripcion']}")
        return True

    if features is None:
        log.warning("Señal presente pero features invalidas — se omite")
        return True

    log.info(f"Señal de estrategia detectada @ {ctx['close']} "
             f"(ATR={ctx['atr']:.2f}) — evaluando con el modelo")

    # Filtrar con el modelo ML.
    prob = inferir(sesion, features)
    log.info(f"Modelo | P(TP)={prob:.3f} | umbral={cfg.umbral_buy}")

    # Verificar entorno del broker.
    ok, motivo = verificar_entorno()
    if not ok:
        log.info(f"Entorno MT5: {motivo}")
        return True

    # Aplicar los filtros de riesgo (ATR, buffer, consistencia, ML).
    ok, motivo = gestor.puede_abrir(ctx["atr"], prob)
    if not ok:
        log.info(f"Riesgo bloquea la operacion: {motivo}")
        return True

    # Abrir la posicion.
    sl, tp = calcular_sl_tp(ctx["close"], ctx["atr"])
    log.info(f"SEÑAL CONFIRMADA | P(TP)={prob:.3f} — abriendo BUY "
             f"(SL={sl} TP={tp})")
    ticket = enviar_orden_buy(ctx["atr"])

    if ticket is not None:
        entrada_real = mt5.symbol_info_tick(cfg.symbol).ask
        pos.abrir(ticket, entrada_real, ahora)
        log.info(f"Posicion abierta | ticket={ticket}")
        r = gestor.resumen()
        log.info(f"Riesgo | balance ${r['balance']:,.2f} | "
                 f"margen DD ${r['margen_dd']:,.2f}")
    else:
        log.error("La orden no se pudo ejecutar")

    return True


# ── Loop principal ────────────────────────────────────────────
async def main():
    configurar_logging()

    if not iniciar_mt5():
        sys.exit(1)

    sesion = cargar_modelo()
    gestor = GestorRiesgo()
    pos    = Posicion()

    # Adoptar una posicion previa del agente si quedo abierta.
    previas = mt5.positions_get(symbol=cfg.symbol)
    if previas:
        for p in previas:
            if p.magic == cfg.magic_number:
                pos.abrir(p.ticket, p.price_open,
                          datetime.fromtimestamp(p.time))
                log.warning(f"Posicion previa adoptada: ticket={p.ticket}")
                break

    log.info(f"Agente activo | intervalo {cfg.intervalo_seg:.0f}s | "
             f"Ctrl+C para detener")

    n = 0
    try:
        while True:
            n += 1
            try:
                seguir = await ciclo(sesion, gestor, pos, n)
                if not seguir:
                    break
            except Exception as e:
                log.error(f"Error inesperado en el ciclo #{n}: {e}",
                          exc_info=True)
            await asyncio.sleep(cfg.intervalo_seg)

    except KeyboardInterrupt:
        log.info("Ctrl+C recibido — deteniendo el agente")

    finally:
        r = resumen_cuenta()
        if r:
            log.info(f"Cuenta final | balance {r['balance']} | "
                     f"equity {r['equity']} | profit {r['profit']}")
        mt5.shutdown()
        log.info("Conexion MT5 cerrada. Agente detenido.")


if __name__ == "__main__":
    asyncio.run(main())
