# Laboratorio — Barridos y comparación de modelos

Notebook de exploración donde probé distintas combinaciones de parámetros
antes de llegar al pipeline final. Esta fue mi parte del TFM: integrar la
estrategia base de Félix con el gestor de riesgo de Marisa, entrenar el
modelo ML y optimizar los parámetros.

## Qué hay acá

- `entrenamientos_barridos_tfm.ipynb` — Barridos de parámetros + comparación de modelos.

## Lo que hice

Dos barridos sobre Donchian, SL/TP y umbral CCI. El segundo agrega filtro
de sesión horaria. En ambos evalué backtesting y forward testing para cada
combinación.

También comparé tres modelos (Logistic Regression, Random Forest, Gradient
Boosting). Random Forest ganó en forward testing con menos overfitting, por
eso quedó en el pipeline final.

## Nota

Usé balance inicial de $10K para que los barridos corrieran rápido en Colab.
Todo es porcentual así que escala directo a la cuenta de $100K del TFM.
