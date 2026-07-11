# =============================================================
# features.py — Ingesta de datos y calculo de features
#
# REGLA DE ORO: los indicadores y las 6 features calculadas aqui
# deben ser IDENTICOS a los del notebook NB_Final_Bot_Trading_TFM
# (celda "Calculo de indicadores tecnicos"). Cualquier diferencia
# produce entradas distintas a las que vio el scaler del modelo y,
# por lo tanto, predicciones incorrectas.
#
# El modelo se sirve con features CRUDAS (sin normalizar): el
# StandardScaler viaja dentro del pipeline_completo.onnx.
# =============================================================

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import logging
from config import cfg

log = logging.getLogger("features")

# Mapa de timeframes de MT5
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


# ── Ingesta desde MT5 ─────────────────────────────────────────
def obtener_rates(symbol: str = None,
                  tf: str = None,
                  n: int = None):
    """
    Descarga las ultimas N velas del timeframe configurado desde MT5.

    Retorna un DataFrame ordenado por tiempo, o None si no hay datos
    suficientes para calcular todas las features.
    """
    symbol = symbol or cfg.symbol
    tf     = tf     or cfg.timeframe
    n      = n      or cfg.n_bars

    rates = mt5.copy_rates_from_pos(symbol, TF_MAP[tf], 0, n)
    if rates is None or len(rates) < cfg.max_60d_ventana + cfg.sma_p:
        log.warning(
            f"Datos insuficientes para {symbol} {tf}: "
            f"{len(rates) if rates is not None else 0} velas | "
            f"{mt5.last_error()}"
        )
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.sort_values("time").reset_index(drop=True)


# ── Indicadores tecnicos (identicos al notebook) ──────────────
def _calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anade al DataFrame los indicadores de la estrategia base:
    SMA200, canal Donchian, ATR de Wilder y CCI. Reproduce
    exactamente la celda de indicadores del notebook.
    """
    df = df.copy()

    # SMA de tendencia
    df["sma_200"] = df["close"].rolling(cfg.sma_p).mean()

    # Canal Donchian con shift(1): la vela actual no entra en su
    # propio maximo/minimo (evita mirar el futuro).
    df["donchian_upper"] = df["high"].rolling(cfg.don_p).max().shift(1)
    df["donchian_lower"] = df["low"].rolling(cfg.don_p).min().shift(1)

    # ATR de Wilder = media exponencial (alpha = 1/n) del True Range
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / cfg.atr_p, adjust=False).mean()

    # CCI = (precio tipico - SMA del precio tipico) / (0.015 x desviacion media)
    tp       = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp   = tp.rolling(cfg.cci_p).mean()
    mean_dev = tp.rolling(cfg.cci_p).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci"] = (tp - sma_tp) / (0.015 * mean_dev)

    # Señal de la estrategia base: las tres condiciones simultaneas
    df["senal"] = (
        (df["close"] > df["donchian_upper"]) &
        (df["close"] > df["sma_200"]) &
        (df["cci"]   > cfg.cci_umb)
    ).astype(int)

    return df


# ── Features de contexto para el modelo ML ────────────────────
def _calcular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anade las 6 features de contexto que consume el modelo.
    Reproduce exactamente la celda de features del notebook.
    """
    df = df.copy()

    # Feature 1: velas consecutivas sobre la SMA200 (comprimidas con log1p).
    sobre = (df["close"] > df["sma_200"]).astype(int).values
    racha = np.zeros(len(df))
    r = 0
    for i in range(len(df)):
        r = r + 1 if sobre[i] == 1 else 0
        racha[i] = r
    df["velas_sobre_sma"] = np.log1p(racha)

    # Feature 2: densidad de señales en las ultimas 520 velas (~20 dias).
    df["densidad_senales"] = df["senal"].rolling(cfg.densidad_ventana).sum()

    # Feature 3: distancia al maximo de 60 dias, en unidades de ATR.
    df["max_60d"]      = df["high"].rolling(cfg.max_60d_ventana).max()
    df["dist_max_60d"] = (df["close"] - df["max_60d"]) / df["atr"]

    # Features 4 y 5: dia de la semana (lunes y viernes tienen peor WR).
    dia = df["time"].dt.dayofweek
    df["es_lunes"]   = (dia == 0).astype(int)
    df["es_viernes"] = (dia == 4).astype(int)

    # Feature 6: hora del dia como funcion ciclica seno.
    df["hora_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)

    return df


# ── Funcion principal ─────────────────────────────────────────
def construir_features(df: pd.DataFrame):
    """
    Calcula indicadores y features sobre la serie descargada y evalua
    la ULTIMA VELA CERRADA (indice -2; la vela -1 aun esta en curso).

    Retorna una tupla:
        (features, hay_senal, contexto)
        - features  : np.ndarray float32 de longitud 6 en el orden de
                      cfg.features, listo para el pipeline ONNX; o None
                      si no se pudieron calcular (NaN o datos insuficientes)
        - hay_senal : True si la vela cerrada cumple las tres condiciones
                      de la estrategia base (Donchian + SMA200 + CCI)
        - contexto  : dict con 'time', 'close' y 'atr' de la vela evaluada
                      (usados por el ejecutor para calcular SL y TP)
    """
    df = _calcular_indicadores(df)
    df = _calcular_features(df)

    # Ultima vela cerrada. La posicion 0 de copy_rates_from_pos es la
    # vela en formacion, por lo que -1 esta sin cerrar y usamos -2.
    if len(df) < 3:
        return None, False, {}
    v = df.iloc[-2]

    contexto = {
        "time":  v["time"],
        "close": float(v["close"]),
        "atr":   float(v["atr"]) if pd.notna(v["atr"]) else np.nan,
    }

    # Señal de la estrategia base sobre la vela cerrada.
    hay_senal = bool(
        (v["close"] > v["donchian_upper"]) and
        (v["close"] > v["sma_200"]) and
        (v["cci"]   > cfg.cci_umb)
    )

    # Vector de features en el orden exacto del scaler.
    valores = v[cfg.features].values.astype(np.float64)

    if np.any(np.isnan(valores)) or np.isnan(contexto["atr"]):
        idx_nan = [cfg.features[k] for k in np.where(np.isnan(valores))[0]]
        log.warning(f"Features con NaN, no se opera este ciclo: {idx_nan}")
        return None, hay_senal, contexto

    return valores.astype(np.float32), hay_senal, contexto


# ── Diagnostico de proximidad (solo informativo) ──────────────
def evaluar_proximidad(df: pd.DataFrame) -> dict:
    """
    Describe que tan cerca esta el mercado de generar una señal.
    No abre ninguna orden: sirve para el log del agente.
    """
    df = _calcular_indicadores(df)
    if len(df) < 3:
        return {"descripcion": "Sin datos suficientes"}
    v = df.iloc[-2]

    sobre_sma  = bool(v["close"] > v["sma_200"])
    sobre_don  = bool(v["close"] > v["donchian_upper"])
    cci_val    = float(v["cci"]) if pd.notna(v["cci"]) else 0.0

    if not sobre_sma:
        desc = "Precio bajo SMA200 — tendencia bajista, sin operar"
    elif not sobre_don:
        desc = "Sobre SMA200 pero sin ruptura del canal Donchian"
    elif cci_val <= cfg.cci_umb:
        desc = f"Ruptura confirmada, falta momentum CCI (actual {cci_val:.0f})"
    else:
        desc = "Las tres condiciones se cumplen — señal activa"

    return {
        "descripcion":  desc,
        "sobre_sma200": sobre_sma,
        "ruptura_don":  sobre_don,
        "cci":          round(cci_val, 1),
        "close":        round(float(v["close"]), 1),
    }
