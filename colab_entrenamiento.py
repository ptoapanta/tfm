# =============================================================
# colab_entrenamiento.py
#
# Ejecuta este archivo en Google Colab, celda por celda.
# Al final obtendrás:
#   - modelo.onnx  → cópialo a la carpeta del agente
#   - Los valores de scaler_mean y scaler_scale para config.py
# =============================================================

# ════════════════════════════════════════════════════════════
# CELDA 1 — Instalar dependencias (solo la primera vez)
# ════════════════════════════════════════════════════════════
# !pip install MetaTrader5 scikit-learn skl2onnx onnxruntime vectorbt TA-Lib -q

# ════════════════════════════════════════════════════════════
# CELDA 2 — Imports
# ════════════════════════════════════════════════════════════
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# ════════════════════════════════════════════════════════════
# CELDA 3 — Conectar a MT5 y descargar datos históricos
# ════════════════════════════════════════════════════════════
# IMPORTANTE: MT5 debe estar abierto en tu PC Windows
# Esta celda solo funciona si ejecutas Colab en local con
# jupyter notebook (ver guía de instalación)
# Alternativa: sube un CSV exportado desde MT5

mt5.initialize()
print(f"MT5 conectado: {mt5.terminal_info().company}")

SYMBOL    = "EURUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
N_BARS    = 10000   # más datos = mejor modelo

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, N_BARS)
df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df.set_index("time", inplace=True)

print(f"Datos descargados: {len(df)} filas")
print(f"Desde: {df.index[0]}  Hasta: {df.index[-1]}")
df.tail(3)

# ════════════════════════════════════════════════════════════
# CELDA 4 — Calcular features (IDÉNTICAS al agente)
# ════════════════════════════════════════════════════════════
# Intentar con TA-Lib, si no está disponible usar pandas
try:
    import talib
    df["sma10"] = talib.SMA(df["close"].values, timeperiod=10)
    df["sma30"] = talib.SMA(df["close"].values, timeperiod=30)
    df["rsi14"] = talib.RSI(df["close"].values, timeperiod=14)
    df["macd"]  = talib.MACD(df["close"].values, 12, 26, 9)[0]
    df["atr14"] = talib.ATR(df["high"].values,
                             df["low"].values,
                             df["close"].values, timeperiod=14)
    print("Features calculadas con TA-Lib")
except ImportError:
    df["sma10"] = df["close"].rolling(10).mean()
    df["sma30"] = df["close"].rolling(30).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - (100 / (1 + gain / loss))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    hl   = df["high"] - df["low"]
    hc   = (df["high"] - df["close"].shift()).abs()
    lc   = (df["low"]  - df["close"].shift()).abs()
    df["atr14"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    print("Features calculadas con pandas (instala TA-Lib para mayor precisión)")

# Retorno y volumen
df["retorno"] = df["close"].pct_change()
df["volume"]  = df["tick_volume"].astype(float)

# Target: 1 si el precio sube en la siguiente vela, 0 si baja
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# Eliminar filas con NaN
df.dropna(inplace=True)
print(f"\nDataset final: {len(df)} filas")
print(f"Distribución del target:\n{df['target'].value_counts()}")

# ════════════════════════════════════════════════════════════
# CELDA 5 — Preparar datos y entrenar modelo
# ════════════════════════════════════════════════════════════
FEATURES = ["sma10", "sma30", "rsi14", "macd", "atr14", "retorno", "volume"]

X = df[FEATURES].values.astype(np.float32)
y = df["target"].values

# División temporal (no aleatoria — importante en series de tiempo)
split = int(len(X) * 0.80)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Normalización
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Entrenar modelo
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
modelo.fit(X_train_sc, y_train)
print("Modelo entrenado")

# ════════════════════════════════════════════════════════════
# CELDA 6 — Evaluar el modelo
# ════════════════════════════════════════════════════════════
y_pred      = modelo.predict(X_test_sc)
y_pred_prob = modelo.predict_proba(X_test_sc)[:, 1]

print("=== Reporte de clasificación ===")
print(classification_report(y_test, y_pred,
                             target_names=["BAJA", "SUBE"]))

auc = roc_auc_score(y_test, y_pred_prob)
print(f"AUC-ROC: {auc:.4f}")
print()
print("NOTA: Un AUC > 0.55 es razonable en mercados financieros.")
print("      No esperes AUC > 0.70 — los mercados son altamente eficientes.")

# ════════════════════════════════════════════════════════════
# CELDA 7 — Exportar a ONNX
# ════════════════════════════════════════════════════════════
n_features   = len(FEATURES)
tipo_entrada = [("float_input", FloatTensorType([None, n_features]))]

onnx_modelo = convert_sklearn(
    modelo,
    initial_types=tipo_entrada,
    target_opset=12,
    options={id(modelo): {"zipmap": False}}   # output como array, no dict
)

with open("modelo.onnx", "wb") as f:
    f.write(onnx_modelo.SerializeToString())

print(f"Modelo exportado: modelo.onnx ({len(onnx_modelo.SerializeToString())} bytes)")

# ════════════════════════════════════════════════════════════
# CELDA 8 — Verificar que ONNX produce los mismos resultados
# ════════════════════════════════════════════════════════════
sesion_onnx  = ort.InferenceSession("modelo.onnx")
nombre_in    = sesion_onnx.get_inputs()[0].name
pred_onnx    = sesion_onnx.run(None, {nombre_in: X_test_sc[:5].astype(np.float32)})

print("=== Verificación ONNX vs Sklearn ===")
print("Sklearn probs :", modelo.predict_proba(X_test_sc[:5]).round(4))
print("ONNX probs    :", pred_onnx[1].round(4))

diff = np.abs(modelo.predict_proba(X_test_sc[:5]) - pred_onnx[1]).max()
print(f"Diferencia max: {diff:.8f}")

if diff < 1e-4:
    print("VALIDACION OK — el modelo ONNX es equivalente al sklearn")
else:
    print("ADVERTENCIA — hay diferencias. Revisa la conversión.")

# ════════════════════════════════════════════════════════════
# CELDA 9 — Copiar estos valores a config.py del agente
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COPIA ESTOS VALORES A config.py DEL AGENTE:")
print("=" * 60)
print(f"scaler_mean  = {scaler.mean_.tolist()}")
print(f"scaler_scale = {scaler.scale_.tolist()}")
print()
print("El archivo modelo.onnx también debe copiarse")
print("a la carpeta del agente en tu PC.")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# CELDA 10 — Backtest rápido con VectorBT (opcional)
# ════════════════════════════════════════════════════════════
# Esta celda muestra cómo habrían funcionado las señales
# del modelo en el período de prueba.

import vectorbt as vbt

# Obtener señales del modelo en el período de test
probs_test = modelo.predict_proba(X_test_sc)[:, 1]

señales_buy  = probs_test >= 0.62
señales_sell = probs_test <= 0.38

# Precios del período de test
precios_test = df["close"].values[split:]

# Backtesting solo señales de compra como ejemplo
pf = vbt.Portfolio.from_signals(
    precios_test,
    entries=señales_buy,
    exits=señales_sell,
    init_cash=10000,
    fees=0.0002,      # 2 pips de comisión
    slippage=0.001,
)

print("=== Backtest del modelo (período de prueba) ===")
print(pf.stats())
