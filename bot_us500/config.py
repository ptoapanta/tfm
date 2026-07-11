# =============================================================
# config.py — Configuracion centralizada del agente US500
#
# Todos los valores de este archivo provienen del proceso de
# optimizacion documentado en el notebook NB_Final_Bot_Trading_TFM
# (celda "PARAMETROS DEFINITIVOS DEL SISTEMA"). Es el unico archivo
# que debe editarse para ajustar el bot antes de ponerlo en marcha.
#
# La estrategia es: ruptura de canal Donchian confirmada por
# tendencia (SMA200) y momentum (CCI), filtrada por un modelo
# Random Forest, y operada bajo las reglas de la cuenta de
# evaluacion GetLeveraged Turbo Trade.
# =============================================================

from pydantic import BaseModel, Field, model_validator
from typing import List


class ConfigAgente(BaseModel):

    # ── Instrumento ───────────────────────────────────────────
    # IMPORTANTE: 'symbol' debe coincidir EXACTAMENTE con el nombre
    # del CFD del S&P 500 en el Market Watch del broker/prop firm.
    # Segun el broker puede ser "US500", "US500Roll", "SP500", etc.
    symbol: str    = "US500"
    timeframe: str = "M15"

    # Numero de velas a descargar en cada ciclo. El calculo de la
    # feature 'dist_max_60d' necesita el maximo de 60 dias
    # (60 x 26 velas M15 = 1560 velas), por lo que este valor debe
    # ser holgadamente superior a 1560 para tener features validas.
    n_bars: int    = Field(3000, ge=1700, le=10000)

    # ── Ritmo del agente ──────────────────────────────────────
    # Cada cuantos segundos el agente revisa si hay una nueva señal.
    # 900 s = 15 min = una vela M15.
    intervalo_seg: float = Field(900.0, ge=60.0)

    # ── Parametros de la estrategia base (celda 3 del notebook) ─
    don_p:   int   = 30     # Canal Donchian: 30 periodos (optimo del barrido)
    sma_p:   int   = 200    # SMA de tendencia: 200 periodos
    cci_p:   int   = 14     # Periodo del CCI
    atr_p:   int   = 14     # Periodo del ATR (Wilder)
    cci_umb: float = 100.0  # Umbral CCI: entrada solo si momentum > 100

    # ── SL y TP en multiplos de ATR ───────────────────────────
    # Deben coincidir con los usados en el etiquetado del notebook.
    sl_atr_mult: float = Field(0.5, ge=0.1, le=5.0)
    tp_atr_mult: float = Field(1.5, ge=0.5, le=10.0)

    # Filtro de volatilidad minima: si el SL resultante es menor
    # a este valor en puntos, el spread se come demasiado margen
    # y la señal se descarta.
    sl_min_pts: float = Field(2.0, ge=0.0)

    # ── Features del modelo (orden EXACTO exigido por el ONNX) ─
    # No modificar el orden: el pipeline_completo.onnx fue exportado
    # con el scaler ajustado a esta secuencia de columnas.
    features: List[str] = [
        "velas_sobre_sma",   # log1p(velas consecutivas sobre SMA200)
        "densidad_senales",  # señales Donchian en las ultimas 520 velas
        "dist_max_60d",      # (close - max 60d) / ATR
        "es_lunes",          # binaria: lunes
        "es_viernes",        # binaria: viernes
        "hora_sin",          # hora del dia como funcion seno
    ]

    # Ventanas de las features (celda 5 del notebook)
    densidad_ventana: int = 520   # 20 dias x 26 velas M15
    max_60d_ventana:  int = 1560  # 60 dias x 26 velas M15

    # ── Modelo ────────────────────────────────────────────────
    # El pipeline ONNX incluye el StandardScaler + Random Forest.
    # El agente pasa las features CRUDAS: el escalado ocurre dentro
    # del propio grafo ONNX. No normalizar manualmente.
    modelo_path: str   = "pipeline_completo.onnx"
    umbral_buy:  float = Field(0.50, ge=0.0, le=1.0)

    # ── Tamaño de posicion ────────────────────────────────────
    # Lote fijo de 22, justificado en el notebook: en el peor
    # escenario historico (6 ops/dia x 11.07 pts de SL x 22 lotes
    # = ~$1.461) la perdida se mantiene por debajo del stop buffer
    # diario de $1.500.
    lote:        float = Field(22.0, ge=0.01, le=100.0)
    valor_punto: float = Field(1.0, ge=0.0)   # $1 por punto por lote en CFD US500

    # ── Gestion de riesgo (GetLeveraged Turbo Trade) ──────────
    balance_inicial:  float = Field(100_000.0, gt=0.0)
    objetivo_pct:     float = Field(0.06, ge=0.0)   # objetivo +6% = $6.000
    dd_pct:           float = Field(0.06, ge=0.0)   # drawdown trailing 6% = $6.000
    limite_dia_pct:   float = Field(0.03, ge=0.0)   # perdida diaria maxima 3% = $3.000
    stop_buf_pct:     float = Field(0.50, ge=0.0)   # el bot para al 50% del limite diario
    consistencia_max: float = Field(0.20, ge=0.0)   # ningun dia > 20% del total acumulado
    max_posiciones:   int   = Field(1, ge=1, le=10)

    # Cierre por tiempo: se cierra la posicion si lleva este numero
    # de velas M15 abierta sin haber tocado TP ni SL (32 velas = 8 h).
    tiempo_max_velas: int = Field(32, ge=1)

    # ── Identificador y registro ──────────────────────────────
    magic_number:    int = 20260101
    log_file:        str = "agente.log"
    operaciones_csv: str = "operaciones.csv"

    # ── Valores derivados (no editar) ─────────────────────────
    @property
    def limite_dia_usd(self) -> float:
        return self.balance_inicial * self.limite_dia_pct

    @property
    def stop_buf_usd(self) -> float:
        return self.limite_dia_usd * self.stop_buf_pct

    @property
    def objetivo_usd(self) -> float:
        return self.balance_inicial * self.objetivo_pct

    @property
    def wr_breakeven(self) -> float:
        # Winrate minimo para no perder dinero dado el ratio SL/TP.
        return self.sl_atr_mult / (self.sl_atr_mult + self.tp_atr_mult) * 100

    # ── Validaciones automaticas ──────────────────────────────
    @model_validator(mode="after")
    def validar_features(self):
        if len(self.features) != 6:
            raise ValueError(
                f"El modelo ONNX espera 6 features y config define "
                f"{len(self.features)}. Revisa la lista 'features'."
            )
        return self

    @model_validator(mode="after")
    def validar_ratio_riesgo(self):
        ratio = self.tp_atr_mult / self.sl_atr_mult
        if ratio < 1.5:
            raise ValueError(
                f"El ratio TP/SL ({ratio:.2f}) es menor a 1.5. "
                f"Con este ratio el breakeven exigiria un winrate "
                f"demasiado alto."
            )
        return self

    @model_validator(mode="after")
    def validar_ventanas(self):
        if self.n_bars <= self.max_60d_ventana:
            raise ValueError(
                f"n_bars ({self.n_bars}) debe ser mayor que "
                f"max_60d_ventana ({self.max_60d_ventana}) para poder "
                f"calcular la feature dist_max_60d."
            )
        return self


# Instancia global — todos los modulos la importan desde aqui.
cfg = ConfigAgente()
