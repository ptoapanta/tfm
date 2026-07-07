# =============================================================
# agente.py — Loop principal del agente de trading US500 v3
#
# CÓMO USAR:
#   1. Abre MetaTrader 5 y conéctate al broker
#   2. Activa Algo Trading en MT5
#   3. Ejecuta: python agente.py
#   4. Para detener: Ctrl+C
# =============================================================

import asyncio
import logging
import sys
import numpy as np
import MetaTrader5 as mt5
import onnxruntime as ort
from datetime import datetime

from config import cfg
from features import obtener_rates, construir_features, normalizar, evaluar_proximidad
from riesgo import puede_operar, resumen_cuenta
from ejecutor import enviar_orden_buy


# ── Configurar logging ────────────────────────────────────────
def configurar_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(cfg.log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


log = logging.getLogger("agente")


# ── Inicializar MT5 ───────────────────────────────────────────
def iniciar_mt5() -> bool:
    if not mt5.initialize():
        log.error(f"No se pudo conectar a MT5: {mt5.last_error()}")
        return False

    # Asegurar que el símbolo está en Market Watch
    if not mt5.symbol_select(cfg.symbol, True):
        log.error(f"No se pudo seleccionar el símbolo {cfg.symbol}")
        return False

    cuenta   = mt5.account_info()
    terminal = mt5.terminal_info()

    log.info("=" * 60)
    log.info("AGENTE US500 v3 — INICIANDO")
    log.info("=" * 60)
    log.info(f"Broker   : {cuenta.company}")
    log.info(f"Cuenta   : {cuenta.login} ({cuenta.currency})")
    log.info(f"Balance  : {cuenta.balance:.2f}")
    log.info(f"Terminal : {terminal.name} | Ruta: {terminal.path}")
    log.info(f"Simbolo  : {cfg.symbol} | TF: {cfg.timeframe}")
    log.info(f"Estrategia: SOLO LONG | EMA{cfg.ema_rapida}/"
             f"EMA{cfg.ema_lenta} + SMA{cfg.sma_tendencia}")
    log.info(f"SL/TP    : {cfg.sl_atr_mult}xATR / {cfg.tp_atr_mult}xATR")
    log.info(f"Lote     : {cfg.lote}")
    log.info(f"Umbral   : {cfg.umbral_buy}")
    log.info(f"Drawdown max: {cfg.max_drawdown_pct:.0%}")
    log.info("=" * 60)
    return True


# ── Cargar modelo ONNX ────────────────────────────────────────
def cargar_modelo() -> ort.InferenceSession:
    try:
        sesion = ort.InferenceSession(
            cfg.modelo_path,
            providers=["CPUExecutionProvider"]
        )
        nombre = sesion.get_inputs()[0].name
        forma  = sesion.get_inputs()[0].shape
        log.info(f"Modelo ONNX cargado: {cfg.modelo_path}")
        log.info(f"Input: {nombre} | Forma: {forma}")
        return sesion
    except FileNotFoundError:
        log.error(
            f"Modelo '{cfg.modelo_path}' no encontrado. "
            f"Copia el archivo .onnx a la carpeta C:\\agente_mt5\\"
        )
        sys.exit(1)
    except Exception as e:
        log.error(f"Error al cargar el modelo ONNX: {e}")
        sys.exit(1)


# ── Inferencia ────────────────────────────────────────────────
def inferir(sesion: ort.InferenceSession,
            features_norm: np.ndarray) -> float:
    """
    Ejecuta el modelo ONNX y retorna la probabilidad
    de que la operación alcance el TP (clase 1).
    """
    nombre  = sesion.get_inputs()[0].name
    entrada = features_norm.reshape(1, -1)
    result  = sesion.run(None, {nombre: entrada})
    # result[1] = [[prob_clase_0, prob_clase_1]]
    return float(result[1][0][1])


# ── Un ciclo completo del agente ──────────────────────────────
async def ciclo(sesion: ort.InferenceSession, n: int) -> None:

    log.info(f"--- Ciclo #{n} | {datetime.now().strftime('%H:%M:%S')} ---")

    # PASO 1: Descargar velas desde MT5
    rates = obtener_rates()
    if rates is None:
        log.warning("Sin datos suficientes — saltando ciclo")
        return

    # PASO 2: Calcular features y detectar señal
    resultado = construir_features(rates)
    if resultado is None or resultado[0] is None:
        log.warning("Error al calcular features — saltando ciclo")
        return

    features, hay_señal = resultado

    if not hay_señal:
        # Evaluar proximidad y registrar el estado
        proximidad = evaluar_proximidad(rates)
        nivel      = proximidad["nivel"]
        iconos     = {0: "  ", 1: "👀", 2: "⚠️ ", 3: "🔔"}
        icono      = iconos.get(nivel, "  ")

        log.info(
            f"{icono} Proximidad nivel {nivel}/3 | "
            f"{proximidad['descripcion']}"
        )
        log.info(
            f"   EMA dist={proximidad['distancia_ema']:+.4f} | "
            f"RSI={proximidad['rsi_actual']} | "
            f"SMA200={'OK' if proximidad['sobre_sma200'] else 'NO'} | "
            f"Pullback={'OK' if proximidad['pullback_ok'] else 'NO'} | "
            f"Convergiendo={'SI' if proximidad['convergiendo'] else 'NO'}"
        )
        return

    log.info("Señal detectada — evaluando con el modelo...")

    # PASO 3: Normalizar features
    try:
        features_norm = normalizar(features)
    except ValueError as e:
        log.error(f"Error al normalizar: {e}")
        return

    # PASO 4: Inferencia del modelo
    prob_tp = inferir(sesion, features_norm)
    log.info(f"Modelo | P(TP)={prob_tp:.3f} | umbral={cfg.umbral_buy}")

    if prob_tp < cfg.umbral_buy:
        log.info(
            f"Probabilidad insuficiente ({prob_tp:.3f} < {cfg.umbral_buy})"
            f" — no operar"
        )
        return

    # PASO 5: Verificar condiciones de riesgo
    ok, motivo = puede_operar()
    if not ok:
        log.info(f"Riesgo: {motivo}")
        return

    # PASO 6: Enviar orden BUY
    log.info(f"SEÑAL CONFIRMADA | P(TP)={prob_tp:.3f} — abriendo BUY")
    exito = enviar_orden_buy()

    if exito:
        log.info("Orden ejecutada correctamente")
    else:
        log.error("La orden no se pudo ejecutar")

    # PASO 7: Registrar estado de la cuenta
    resumen = resumen_cuenta()
    if resumen:
        log.info(
            f"Cuenta | Balance={resumen['balance']} | "
            f"Equity={resumen['equity']} | "
            f"Profit={resumen['profit']} | "
            f"Drawdown={resumen['drawdown']}%"
        )


# ── Loop infinito ─────────────────────────────────────────────
async def main():
    configurar_logging()

    if not iniciar_mt5():
        sys.exit(1)

    sesion  = cargar_modelo()
    n_ciclo = 0

    log.info(
        f"Agente activo | Intervalo: {cfg.intervalo_seg}s | "
        f"Ctrl+C para detener"
    )

    try:
        while True:
            n_ciclo += 1
            try:
                await ciclo(sesion, n_ciclo)
            except Exception as e:
                log.error(
                    f"Error inesperado en ciclo #{n_ciclo}: {e}",
                    exc_info=True
                )
            await asyncio.sleep(cfg.intervalo_seg)

    except KeyboardInterrupt:
        log.info("Ctrl+C recibido — deteniendo el agente...")

    finally:
        mt5.shutdown()
        log.info("Conexión MT5 cerrada. Agente detenido.")


# ── Punto de entrada ──────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())