# =============================================================
# features.py — Ingesta de datos y features para el modelo v3
#
# REGLA DE ORO: Estas 16 features deben ser IDÉNTICAS
# a las calculadas en el notebook US500_MA_v3_Colab.ipynb.
# Cualquier diferencia produce predicciones incorrectas.
# =============================================================

import numpy as np
import MetaTrader5 as mt5
import logging
from config import cfg

log = logging.getLogger("features")

# Mapa de timeframes
TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


# ── Funciones de indicadores (idénticas al notebook) ──────────

def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    """EMA con adjust=False — igual que pandas ewm(span=n, adjust=False)."""
    alpha  = 2.0 / (n + 1)
    result = np.empty(len(arr))
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _sma(arr: np.ndarray, n: int) -> np.ndarray:
    """SMA estándar."""
    result = np.full(len(arr), np.nan)
    for i in range(n - 1, len(arr)):
        result[i] = np.mean(arr[i - n + 1: i + 1])
    return result


def _rsi_wilder(arr: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI de Wilder — igual que pandas ewm(alpha=1/n, adjust=False)."""
    delta  = np.diff(arr)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    alpha  = 1.0 / n
    avg_g  = np.empty(len(gains))
    avg_l  = np.empty(len(losses))
    avg_g[0] = gains[0]
    avg_l[0] = losses[0]

    for i in range(1, len(gains)):
        avg_g[i] = alpha * gains[i]  + (1 - alpha) * avg_g[i - 1]
        avg_l[i] = alpha * losses[i] + (1 - alpha) * avg_l[i - 1]

    rsi    = np.full(len(arr), np.nan)
    for i in range(len(avg_g)):
        if avg_l[i] == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + avg_g[i] / avg_l[i]))
    return rsi


def _atr_wilder(high: np.ndarray, low: np.ndarray,
                close: np.ndarray, n: int = 14) -> np.ndarray:
    """ATR de Wilder — igual que pandas ewm(alpha=1/n, adjust=False)."""
    tr     = np.maximum(high[1:] - low[1:],
             np.maximum(np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:]  - close[:-1])))
    alpha  = 1.0 / n
    atr    = np.empty(len(tr))
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    # Alinear con el array original (un NaN al inicio)
    result    = np.full(len(close), np.nan)
    result[1:] = atr
    return result


# ── Función principal de ingesta ──────────────────────────────

def obtener_rates(symbol: str = cfg.symbol,
                  tf: str     = cfg.timeframe,
                  n: int      = cfg.n_bars) -> np.ndarray | None:
    """
    Descarga las últimas N velas M15 desde MT5.
    Retorna structured array numpy o None si falla.
    """
    rates = mt5.copy_rates_from_pos(symbol, TF_MAP[tf], 0, n)
    if rates is None or len(rates) < 400:
        log.warning(f"Datos insuficientes para {symbol} {tf}: "
                    f"{mt5.last_error()}")
        return None
    return rates


# ── Detección de señal y cálculo de features ─────────────────

def construir_features(rates: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """
    Analiza las últimas velas y determina si hay una señal válida.

    Retorna:
        (features_array, hay_señal)
        - features_array : array float32 de 16 features listo para ONNX
        - hay_señal      : True si se cumplen los tres filtros de entrada

    Filtros requeridos para hay_señal = True:
        1. Cruce alcista EMA9 sobre EMA21 en la última vela cerrada
        2. Precio sobre SMA200 (tendencia alcista confirmada)
        3. RSI estuvo por debajo de rsi_pullback en las últimas
           lookback_rsi velas (pullback previo)
    """
    close = rates["close"].astype(np.float64)
    high  = rates["high"].astype(np.float64)
    low   = rates["low"].astype(np.float64)
    open_ = rates["open"].astype(np.float64)

    n = len(close)

    # ── Indicadores base ──────────────────────────────────────
    ema_r  = _ema(close, cfg.ema_rapida)          # EMA9
    ema_l  = _ema(close, cfg.ema_lenta)           # EMA21
    sma_t  = _sma(close, cfg.sma_tendencia)       # SMA200
    sma50  = _sma(close, 50)
    rsi14  = _rsi_wilder(close, cfg.rsi_periodo)  # RSI14
    atr14  = _atr_wilder(high, low, close, cfg.atr_periodo)

    # Contexto H4 (EMA9*16 y EMA21*16 en M15 equivalen a H4)
    ema_r_H4 = _ema(close, cfg.ema_rapida * 16)
    ema_l_H4 = _ema(close, cfg.ema_lenta  * 16)
    rsi_H4   = _rsi_wilder(close, cfg.rsi_periodo * 4)

    # ── Verificar que hay suficientes datos ───────────────────
    min_idx = max(cfg.sma_tendencia, cfg.ema_lenta * 16,
                  cfg.rsi_periodo * 4) + cfg.lookback_rsi + 10
    if n < min_idx:
        log.warning(f"Velas insuficientes: {n} < {min_idx}")
        return None, False

    # ── Detectar señal en la última vela cerrada (índice -2) ──
    # Usamos -2 porque -1 es la vela en curso (no cerrada aún)
    i = -2

    # Filtro 1: Cruce alcista EMA9 sobre EMA21
    cruce_alc = (ema_r[i] > ema_l[i]) and (ema_r[i - 1] <= ema_l[i - 1])

    # Filtro 2: Precio sobre SMA200
    sobre_tendencia = close[i] > sma_t[i]

    # Filtro 3: RSI estuvo bajo rsi_pullback en las últimas lookback_rsi velas
    rsi_ventana = rsi14[i - cfg.lookback_rsi: i]
    rsi_validos = rsi_ventana[~np.isnan(rsi_ventana)]
    pullback_previo = (len(rsi_validos) > 0 and
                       np.min(rsi_validos) < cfg.rsi_pullback)

    hay_señal = cruce_alc and sobre_tendencia and pullback_previo

    # ── Calcular las 16 features (siempre, señal o no) ───────
    # Normalizadas por ATR para comparabilidad temporal
    atr_v = atr14[i]
    if np.isnan(atr_v) or atr_v == 0:
        log.warning("ATR inválido")
        return None, False

    # Retornos
    ret4  = (close[i] - close[i - 4])  / close[i - 4]  if close[i-4]  != 0 else 0.0
    ret16 = (close[i] - close[i - 16]) / close[i - 16] if close[i-16] != 0 else 0.0
    ret64 = (close[i] - close[i - 64]) / close[i - 64] if close[i-64] != 0 else 0.0

    # Volatilidad últimas 10 velas
    rets_10 = np.diff(close[i - 11: i + 1]) / close[i - 11: i]
    vol10   = np.std(rets_10) if len(rets_10) > 1 else 0.0

    # RSI mínimo en las últimas lookback_rsi velas
    rsi_min = (np.nanmin(rsi14[i - cfg.lookback_rsi: i])
               if not np.all(np.isnan(rsi14[i - cfg.lookback_rsi: i]))
               else 50.0)

    features = np.array([
        # [0]  Fuerza del cruce M15
        (ema_r[i] - ema_l[i]) / atr_v,
        # [1]  Aceleración EMA rápida (pendiente 3 velas)
        (ema_r[i] - ema_r[i - 3]) / atr_v,
        # [2]  Dirección EMA lenta (pendiente 3 velas)
        (ema_l[i] - ema_l[i - 3]) / atr_v,
        # [3]  Distancia precio - SMA50
        (close[i] - sma50[i]) / atr_v,
        # [4]  Distancia precio - SMA200 (profundidad pullback)
        (close[i] - sma_t[i]) / atr_v,
        # [5]  RSI M15 normalizado
        (rsi14[i] - 50.0) / 50.0,
        # [6]  RSI mínimo previo normalizado
        (rsi_min - 50.0) / 50.0,
        # [7]  RSI H4 normalizado
        (rsi_H4[i] - 50.0) / 50.0 if not np.isnan(rsi_H4[i]) else 0.0,
        # [8]  Alineación H4
        (ema_r_H4[i] - ema_l_H4[i]) / atr_v,
        # [9]  Cuerpo de la vela del cruce
        (close[i] - open_[i]) / atr_v,
        # [10] Rango de la vela del cruce
        (high[i] - low[i]) / atr_v,
        # [11] Momentum 1h (4 velas M15)
        ret4,
        # [12] Momentum 4h (16 velas M15)
        ret16,
        # [13] Momentum 16h (64 velas M15)
        ret64,
        # [14] Volatilidad últimas 10 velas
        vol10,
        # [15] ATR relativo al precio (régimen de volatilidad)
        atr_v / sma50[i] * 100.0 if sma50[i] != 0 else 0.0,
    ], dtype=np.float64)

    # ── Verificar NaN ─────────────────────────────────────────
    if np.any(np.isnan(features)):
        nan_idx = np.where(np.isnan(features))[0]
        log.warning(f"NaN en features índices: {nan_idx}")
        return None, False

    return features.astype(np.float32), hay_señal


# ── Normalización ─────────────────────────────────────────────

def normalizar(features: np.ndarray) -> np.ndarray:
    """
    Aplica StandardScaler con los parámetros de config.py.
    Equivalente a scaler.transform(features.reshape(1, -1)).
    """
    mean  = np.array(cfg.scaler_mean,  dtype=np.float32)
    scale = np.array(cfg.scaler_scale, dtype=np.float32)

    if len(features) != len(mean):
        raise ValueError(
            f"features tiene {len(features)} valores "
            f"pero scaler_mean tiene {len(mean)}. "
            f"Verifica que config.py y features.py están sincronizados."
        )

    return ((features - mean) / scale).astype(np.float32)



# ── Detección de proximidad a una señal ───────────────────────

def evaluar_proximidad(rates: np.ndarray) -> dict:
    """
    Evalúa qué tan cerca está el mercado de generar una señal.
    No ejecuta ninguna orden — solo informa el estado actual.

    Retorna un diccionario con:
        - nivel        : 0=lejos, 1=observar, 2=cerca, 3=inminente
        - descripcion  : texto explicativo
        - distancia_ema: diferencia EMA9-EMA21 normalizada por ATR
        - rsi_actual   : RSI actual
        - sobre_sma200 : True si precio está sobre SMA200
        - pullback_ok  : True si RSI estuvo bajo el nivel de pullback
    """
    close = rates["close"].astype(np.float64)
    high  = rates["high"].astype(np.float64)
    low   = rates["low"].astype(np.float64)

    ema_r  = _ema(close, cfg.ema_rapida)
    ema_l  = _ema(close, cfg.ema_lenta)
    sma_t  = _sma(close, cfg.sma_tendencia)
    rsi14  = _rsi_wilder(close, cfg.rsi_periodo)
    atr14  = _atr_wilder(high, low, close, cfg.atr_periodo)

    i = -2  # última vela cerrada

    atr_v        = atr14[i] if not np.isnan(atr14[i]) else 1.0
    dist_ema      = (ema_r[i] - ema_l[i]) / atr_v
    dist_ema_prev = (ema_r[i-1] - ema_l[i-1]) / atr_v
    rsi_actual    = rsi14[i] if not np.isnan(rsi14[i]) else 50.0
    sobre_sma200  = bool(close[i] > sma_t[i])

    # Pullback: RSI estuvo bajo el nivel en las últimas velas
    rsi_ventana  = rsi14[i - cfg.lookback_rsi: i]
    rsi_validos  = rsi_ventana[~np.isnan(rsi_ventana)]
    pullback_ok  = (len(rsi_validos) > 0 and
                    np.min(rsi_validos) < cfg.rsi_pullback)

    # ── Convergencia: las EMAs se están acercando ─────────────
    convergiendo = (dist_ema < 0) and (dist_ema > dist_ema_prev)

    # ── Niveles de proximidad ─────────────────────────────────
    if not sobre_sma200:
        nivel       = 0
        descripcion = "Precio bajo SMA200 — tendencia bajista, sin operar"

    elif dist_ema > 1.0:
        nivel       = 0
        descripcion = "EMAs muy separadas — lejos de un cruce"

    elif dist_ema > 0.3:
        nivel       = 1
        descripcion = "EMAs separadas — observar"

    elif dist_ema > 0 and not convergiendo:
        nivel       = 1
        descripcion = "EMA9 sobre EMA21 pero sin convergencia"

    elif convergiendo and not pullback_ok:
        nivel       = 2
        descripcion = (
            "EMAs convergiendo hacia cruce alcista — "
            "falta pullback del RSI"
        )

    elif convergiendo and pullback_ok:
        nivel       = 3
        descripcion = (
            "CRUCE INMINENTE — EMAs convergiendo + "
            "pullback confirmado + tendencia alcista"
        )

    elif dist_ema <= 0 and pullback_ok and sobre_sma200:
        nivel       = 2
        descripcion = (
            "EMA9 bajo EMA21 con pullback activo — "
            "posible cruce próximo"
        )
    else:
        nivel       = 1
        descripcion = "Condiciones parciales — seguir monitoreando"

    return {
        "nivel":         nivel,
        "descripcion":   descripcion,
        "distancia_ema": round(float(dist_ema), 4),
        "rsi_actual":    round(float(rsi_actual), 1),
        "sobre_sma200":  sobre_sma200,
        "pullback_ok":   pullback_ok,
        "convergiendo":  convergiendo,
    }