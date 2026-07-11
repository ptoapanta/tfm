# Bot de Trading Algorítmico US500 — TFM

Bot de trading automatizado sobre el CFD del índice S&P 500 (US500), operado en
MetaTrader 5 mediante Python. El sistema implementa una **arquitectura desacoplada
de doble estrategia**: una capa genera señales de rentabilidad y otra, independiente,
valida cada señal contra las reglas de una cuenta de fondeo (GetLeveraged Turbo Trade).

Este repositorio contiene el código final derivado del notebook de investigación
`NB_Final_Bot_Trading_TFM`, donde se documenta el proceso completo de optimización.

## Estrategia

Se abre una operación **long** únicamente cuando se cumplen las tres condiciones de
la estrategia base sobre una vela M15 cerrada:

1. Ruptura del canal **Donchian** de 30 períodos (`close > banda superior`).
2. Tendencia alcista confirmada (`close > SMA 200`).
3. Momentum positivo (`CCI(14) > 100`).

Cada señal se filtra con un modelo **Random Forest** entrenado sobre las señales
históricas de la estrategia. El stop loss se fija en 0.5×ATR y el take profit en
1.5×ATR (ratio 3:1). Solo se ejecuta la operación si la probabilidad estimada por
el modelo supera 0.50 y si el gestor de riesgo lo autoriza.

## Arquitectura

El código se organiza en módulos por responsabilidad, reflejando la separación entre
la lógica de rentabilidad y la de supervivencia:

| Archivo | Responsabilidad |
|---|---|
| `config.py` | Configuración centralizada (todos los parámetros del sistema). |
| `features.py` | Ingesta de datos desde MT5, indicadores y las 6 features del modelo. |
| `riesgo.py` | Gestor de riesgo con estado: reglas de la cuenta de fondeo. |
| `ejecutor.py` | Envío de órdenes al broker y registro de operaciones. |
| `agente.py` | Loop principal que integra los componentes. |
| `colab_entrenamiento.py` | Reproduce el entrenamiento y regenera el modelo ONNX. |
| `pipeline_completo.onnx` | StandardScaler + Random Forest exportados juntos. |

**Modelo en un único ONNX:** el escalado (`StandardScaler`) viaja dentro del mismo
grafo que el Random Forest. El agente pasa las features **crudas** y el escalado
ocurre dentro del ONNX. Esto elimina la posibilidad de que el scaler del código se
desincronice del scaler con el que se entrenó el modelo, y hace el artefacto portable
a otros entornos (MT5 nativo, C++, .NET).

## Las 6 features (orden exacto exigido por el modelo)

1. `velas_sobre_sma` — `log1p` de las velas consecutivas sobre la SMA200.
2. `densidad_senales` — número de señales en las últimas 520 velas (~20 días).
3. `dist_max_60d` — distancia al máximo de 60 días, en unidades de ATR.
4. `es_lunes` — variable binaria (lunes tiene peor winrate histórico).
5. `es_viernes` — variable binaria (viernes tiene peor winrate histórico).
6. `hora_sin` — hora del día como función seno (ciclo de 24 h).

## Reglas de riesgo (GetLeveraged Turbo Trade)

Cuenta de evaluación de $100.000. El agente detiene la operativa si alcanza el
objetivo o si excede el drawdown, y descarta señales individuales según los filtros:

- **Objetivo:** +6% ($6.000) → se detiene el agente.
- **Drawdown trailing:** 6% sobre el balance realizado → se detiene el agente.
- **Stop buffer diario:** pérdida del día ≥ $1.500 (50% del límite diario) → no opera.
- **Consistencia:** ningún día debe concentrar más del 20% de la ganancia total.
- **Volatilidad mínima:** si SL < 2 puntos, la señal se descarta.
- **Tamaño de posición:** lote fijo de 22.

## Uso

### Ejecución del agente

1. Abrir MetaTrader 5 y conectarse al broker o prop firm.
2. Activar el trading algorítmico en MT5.
3. Ajustar `symbol` en `config.py` al nombre exacto del CFD en el Market Watch.
4. Colocar `pipeline_completo.onnx` en la carpeta del agente.
5. Ejecutar:

```bash
pip install -r requirements.txt
python agente.py
```

### Reentrenamiento del modelo

Con un CSV de velas M15 exportado desde MT5:

```bash
python colab_entrenamiento.py
```

Genera un nuevo `pipeline_completo.onnx` y verifica su equivalencia con el modelo
de scikit-learn.

## Instrumento y datos

- Instrumento: CFD US500 (réplica del S&P 500).
- Timeframe: M15.
- Fuente de datos: MetaTrader 5.
