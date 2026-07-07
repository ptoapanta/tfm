# =============================================================
# config.py — Configuración centralizada del agente US500 v3
# Este es el ÚNICO archivo que debes editar para ajustar el bot
# =============================================================

from pydantic import BaseModel, Field, model_validator
from typing import List


class ConfigAgente(BaseModel):

    # ── Instrumento ───────────────────────────────────────────
    symbol: str    = "US500Roll"
    timeframe: str = "M15"
    n_bars: int    = Field(500, ge=100, le=5000)

    # ── Ritmo del agente ──────────────────────────────────────
    # Cada cuántos segundos el agente revisa si hay señal
    # 900 segundos = 15 minutos (equivale a 1 vela M15)
    intervalo_seg: float = Field(900.0, ge=60.0)

    # ── Tamaño de posición ────────────────────────────────────
    lote: float = Field(0.01, ge=0.01, le=100.0)

    # ── SL y TP en múltiplos de ATR ───────────────────────────
    # Deben coincidir con los usados en el entrenamiento (Celda 4 del notebook)
    sl_atr_mult: float = Field(1.5, ge=0.5, le=5.0)
    tp_atr_mult: float = Field(3.0, ge=1.0, le=10.0)

    # ── Parámetros de la señal (deben coincidir con features.py) ──
    ema_rapida:    int   = 9     # períodos EMA rápida en M15
    ema_lenta:     int   = 21    # períodos EMA lenta en M15
    sma_tendencia: int   = 200   # períodos SMA de tendencia en M15
    rsi_periodo:   int   = 14
    rsi_pullback:  float = 45.0  # RSI debe haber estado bajo este nivel
    lookback_rsi:  int   = 8     # velas de lookback para el pullback
    atr_periodo:   int   = 14

    # ── Umbral del modelo ─────────────────────────────────────
    # El agente abre BUY solo si prob >= umbral_buy
    # umbral_sell en 99.0 significa NUNCA operar en corto en el US500
    umbral_buy:  float = Field(0.38, ge=0.0, le=1.0)
    umbral_sell: float = Field(99.0)   # deshabilitado intencionalmente

    # ── Gestión de riesgo ─────────────────────────────────────
    max_posiciones:    int   = Field(1, ge=1, le=10)
    max_drawdown_pct:  float = Field(0.05, ge=0.01, le=0.50)  # 5% del balance

    # ── Modelo ONNX ───────────────────────────────────────────
    modelo_path: str = "modelo_us500_v3.onnx"

    # ── Parámetros del StandardScaler ────────────────────────
    # IMPORTANTE: copia estos valores desde la Celda 9 del notebook v3
    # Formato: la lista completa que imprimió scaler.mean_ y scaler.scale_
    scaler_mean:  List[float] = [
        0.08609,   # f_diff_ema
        0.49741,   # f_slope_r
        0.18319,   # f_slope_l
        0.97681,   # f_dist_sma50
        4.94272,   # f_dist_sma200
        0.17369,   # f_rsi14
        -0.23083,   # f_rsi_min
        0.08842,   # f_rsi_H4
        3.66153,   # f_diff_H4
        0.75587,   # f_cuerpo
        1.36003,   # f_rango
        0.00174,   # f_ret4
        0.00103,   # f_ret16
        0.00165,   # f_ret64
        0.00075,   # f_vol10
        0.09843,   # f_vol_rel
    ]
    scaler_scale: List[float] = [
        0.08258,   # f_diff_ema
        0.32926,   # f_slope_r
        0.12579,   # f_slope_l
        1.00631,   # f_dist_sma50
        4.18904,   # f_dist_sma200
        0.103,   # f_rsi14
        0.10259,   # f_rsi_min
        0.06237,   # f_rsi_H4
        4.1782,   # f_diff_H4
        0.92975,   # f_cuerpo
        1.00623,   # f_rango
        0.00185,   # f_ret4
        0.00171,   # f_ret16
        0.00353,   # f_ret64
        0.00063,   # f_vol10
        0.07083,   # f_vol_rel
    ]

    # ── Identificador del agente ──────────────────────────────
    magic_number: int = 20260101

    # ── Archivos de registro ──────────────────────────────────
    log_file:        str = "agente.log"
    operaciones_csv: str = "operaciones.csv"

    # ── Validaciones automáticas (no editar) ─────────────────
    @model_validator(mode="after")
    def validar_scaler(self):
        if len(self.scaler_mean) != len(self.scaler_scale):
            raise ValueError(
                f"scaler_mean tiene {len(self.scaler_mean)} valores "
                f"pero scaler_scale tiene {len(self.scaler_scale)}. "
                f"Deben tener la misma longitud (16)."
            )
        return self

    @model_validator(mode="after")
    def validar_ratio_riesgo(self):
        ratio = self.tp_atr_mult / self.sl_atr_mult
        if ratio < 1.5:
            raise ValueError(
                f"El ratio TP/SL ({ratio:.1f}) es menor a 1.5. "
                f"Aumenta tp_atr_mult o reduce sl_atr_mult."
            )
        return self


# Instancia global — todos los módulos la importan desde aquí
cfg = ConfigAgente()