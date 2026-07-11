# =============================================================
# colab_entrenamiento.py
#
# Reproduce el entrenamiento del modelo tal como esta documentado
# en el notebook NB_Final_Bot_Trading_TFM. Ejecutando este script
# se regenera el artefacto que consume el agente:
#
#     pipeline_completo.onnx   (StandardScaler + Random Forest)
#
# El flujo es identico al del notebook: carga del CSV, calculo de
# indicadores y las 6 features, etiquetado por simulacion de TP/SL,
# division temporal, entrenamiento del Random Forest sobre las
# señales de la estrategia y exportacion a ONNX.
#
# Puede ejecutarse como script (python colab_entrenamiento.py) o
# copiarse por bloques en celdas de Google Colab.
# =============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort


# =============================================================
# 1. Parametros (identicos a config.py y al notebook)
# =============================================================
RUTA_CSV = "datos_ohlc_US500.csv"   # CSV M15 exportado desde MT5

DON_P   = 30      # canal Donchian
SMA_P   = 200     # SMA de tendencia
CCI_P   = 14      # periodo CCI
ATR_P   = 14      # periodo ATR (Wilder)
CCI_UMB = 100     # umbral de momentum
SL_MULT = 0.5     # stop loss = 0.5 x ATR
TP_MULT = 1.5     # take profit = 1.5 x ATR
LOOK    = 30      # velas de lookahead para el etiquetado

UMBRAL_ML = 0.50

FEATURES = [
    "velas_sobre_sma",
    "densidad_senales",
    "dist_max_60d",
    "es_lunes",
    "es_viernes",
    "hora_sin",
]

FECHA_FIN_TRAIN = "2025-09-30"   # train: mar 2023 -> sep 2025
FECHA_FIN_BT    = "2025-12-31"   # backtest: oct 2025 -> dic 2025
# forward: ene 2026 -> jul 2026 (datos nunca vistos por el modelo)

DENSIDAD_VENTANA = 520    # 20 dias x 26 velas M15
MAX_60D_VENTANA  = 1560   # 60 dias x 26 velas M15

WR_BREAKEVEN = SL_MULT / (SL_MULT + TP_MULT) * 100   # 25%


# =============================================================
# 2. Carga de datos
# =============================================================
df = pd.read_csv(RUTA_CSV)
df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)
print(f"Dataset cargado: {len(df):,} velas | "
      f"{df['time'].min()} -> {df['time'].max()}")


# =============================================================
# 3. Indicadores tecnicos
# =============================================================
df["sma_200"] = df["close"].rolling(SMA_P).mean()

df["donchian_upper"] = df["high"].rolling(DON_P).max().shift(1)
df["donchian_lower"] = df["low"].rolling(DON_P).min().shift(1)

hl = df["high"] - df["low"]
hc = (df["high"] - df["close"].shift(1)).abs()
lc = (df["low"]  - df["close"].shift(1)).abs()
df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(
    alpha=1 / ATR_P, adjust=False).mean()

tp       = (df["high"] + df["low"] + df["close"]) / 3
sma_tp   = tp.rolling(CCI_P).mean()
mean_dev = tp.rolling(CCI_P).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
df["cci"] = (tp - sma_tp) / (0.015 * mean_dev)

df["senal"] = (
    (df["close"] > df["donchian_upper"]) &
    (df["close"] > df["sma_200"]) &
    (df["cci"]   > CCI_UMB)
).astype(int)


# =============================================================
# 4. Features de contexto (las 6 que consume el modelo)
# =============================================================
# Feature 1: velas consecutivas sobre la SMA200 (log1p)
sobre = (df["close"] > df["sma_200"]).astype(int).values
racha = np.zeros(len(df)); r = 0
for i in range(len(df)):
    r = r + 1 if sobre[i] == 1 else 0
    racha[i] = r
df["velas_sobre_sma"] = np.log1p(racha)

# Feature 2: densidad de señales en las ultimas 520 velas
df["densidad_senales"] = df["senal"].rolling(DENSIDAD_VENTANA).sum()

# Feature 3: distancia al maximo de 60 dias, en ATRs
df["max_60d"]      = df["high"].rolling(MAX_60D_VENTANA).max()
df["dist_max_60d"] = (df["close"] - df["max_60d"]) / df["atr"]

# Features 4 y 5: dia de la semana
dia = df["time"].dt.dayofweek
df["es_lunes"]   = (dia == 0).astype(int)
df["es_viernes"] = (dia == 4).astype(int)

# Feature 6: hora del dia como funcion seno
df["hora_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)

df = df.dropna().reset_index(drop=True)
print(f"Indicadores y features calculados. Dataset limpio: {len(df):,} velas")
print(f"Señales generadas por la estrategia: {df['senal'].sum():,}")


# =============================================================
# 5. Etiquetado por simulacion de TP y SL
# =============================================================
# Para cada vela se proyectan TP y SL y se recorren las siguientes
# LOOK velas. La etiqueta es 1 si se toca antes el TP, 0 si el SL.
# Las velas ambiguas (tocan ambos) se resuelven por proximidad del open.
print("Calculando etiquetas...")
cerrar  = df["close"].values
abre    = df["open"].values
maximos = df["high"].values
minimos = df["low"].values
atrs    = df["atr"].values
n       = len(df)
etiq    = np.zeros(n, dtype=int)
ambiguas = 0

for i in range(n - LOOK):
    pe  = cerrar[i]
    atr = atrs[i]
    ntp = pe + atr * TP_MULT
    nsl = pe - atr * SL_MULT
    for j in range(i + 1, i + LOOK + 1):
        toca_tp = maximos[j] >= ntp
        toca_sl = minimos[j] <= nsl
        if toca_tp and toca_sl:
            ambiguas += 1
            etiq[i] = 1 if abs(abre[j] - ntp) < abs(abre[j] - nsl) else 0
            break
        elif toca_tp:
            etiq[i] = 1
            break
        elif toca_sl:
            etiq[i] = 0
            break

df["etiqueta"] = etiq
señales_df = df[df["senal"] == 1]
print(f"Etiquetado listo. Ambiguas: {ambiguas:,} | "
      f"WR natural de las señales: {señales_df['etiqueta'].mean()*100:.1f}% "
      f"(breakeven {WR_BREAKEVEN:.1f}%)")


# =============================================================
# 6. Division temporal
# =============================================================
df_train = df[df["time"] <= FECHA_FIN_TRAIN].copy()
df_bt    = df[(df["time"] > FECHA_FIN_TRAIN) & (df["time"] <= FECHA_FIN_BT)].copy()
df_fw    = df[df["time"] > FECHA_FIN_BT].copy()

s_train = df_train[df_train["senal"] == 1].copy()
s_bt    = df_bt[df_bt["senal"] == 1].copy()
s_fw    = df_fw[df_fw["senal"] == 1].copy()

print(f"Señales -> train: {len(s_train)} | bt: {len(s_bt)} | fw: {len(s_fw)}")


# =============================================================
# 7. Entrenamiento del modelo
# =============================================================
# El scaler se ajusta SOLO con datos de entrenamiento para evitar
# data leakage. El modelo se entrena sobre las señales de la
# estrategia base (no sobre todas las velas).
scaler = StandardScaler()
scaler.fit(s_train[FEATURES])

X_train = scaler.transform(s_train[FEATURES]); y_train = s_train["etiqueta"]
X_bt    = scaler.transform(s_bt[FEATURES]);    y_bt    = s_bt["etiqueta"]
X_fw    = scaler.transform(s_fw[FEATURES]);    y_fw    = s_fw["etiqueta"]

modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=8,
    random_state=42,
    class_weight="balanced",
)
modelo.fit(X_train, y_train)

auc_train = roc_auc_score(y_train, modelo.predict_proba(X_train)[:, 1])
auc_bt    = roc_auc_score(y_bt,    modelo.predict_proba(X_bt)[:, 1])
auc_fw    = roc_auc_score(y_fw,    modelo.predict_proba(X_fw)[:, 1])
print(f"AUC | train: {auc_train:.4f} | bt: {auc_bt:.4f} | "
      f"fw: {auc_fw:.4f} (el que importa)")

# Validacion del poder discriminativo del filtro en forward testing.
probs_fw   = modelo.predict_proba(X_fw)[:, 1]
mask       = probs_fw >= UMBRAL_ML
wr_pasan   = y_fw[mask].mean() * 100
wr_nopasan = y_fw[~mask].mean() * 100
print(f"Forward | WR señales que PASAN el ML: {wr_pasan:.1f}% | "
      f"BLOQUEADAS: {wr_nopasan:.1f}% | diferencia {wr_pasan-wr_nopasan:+.1f}pp")


# =============================================================
# 8. Exportacion a ONNX (scaler + Random Forest en un pipeline)
# =============================================================
# Se exporta el pipeline completo para que el escalado viaje dentro
# del propio grafo ONNX: el agente pasa las features crudas y el
# modelo no puede desincronizarse del scaler.
n_features   = len(FEATURES)
tipo_entrada = [("float_input", FloatTensorType([None, n_features]))]

pipeline = Pipeline([("scaler", scaler), ("modelo", modelo)])
pipeline_onnx = convert_sklearn(pipeline, initial_types=tipo_entrada,
                                target_opset=12)

with open("pipeline_completo.onnx", "wb") as f:
    f.write(pipeline_onnx.SerializeToString())

# Verificacion de equivalencia sklearn vs ONNX.
muestra   = s_fw[FEATURES].values[:5].astype(np.float32)
p_sklearn = modelo.predict_proba(scaler.transform(muestra))[:, 1]

sess    = ort.InferenceSession("pipeline_completo.onnx")
nombre  = sess.get_inputs()[0].name
salida  = sess.run(None, {nombre: muestra})
p_onnx  = np.array([p[1] for p in salida[1]])

diff = np.abs(p_sklearn - p_onnx).max()
print(f"Verificacion ONNX | diferencia maxima: {diff:.2e} "
      f"({'OK' if diff < 1e-5 else 'REVISAR'})")
print("Artefacto exportado: pipeline_completo.onnx")
print(f"Features en orden: {FEATURES}")
