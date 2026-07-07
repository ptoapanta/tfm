TFM
Arquitectura
El diseño del agente utiliza un modelo híbrico de ejecución. Por un lado el modelo se entrena y genera con sus parámetros correspondientes en Google Colab. 
Localmente se ejecuta el agente, a partir del modelo generado en Google Colab (modelo.onnx). También se utilizan archivos de configuración y riesgo 
para controlar las reglas de trading del broker. Finalmente, el agente, a partir del modelo entrenado ,las reglas de gestión de riesgo configuradas y 
la estrategia de trading analiza en ciclos de 15 minutos (temporalidad M15 en MetaTrader5) la ejecución de operaciones de trading.

En esta imagen se describen cada uno de los componentes que forman parte de la solución propuesta para el TFM.
<img width="1440" height="2530" alt="image" src="https://github.com/user-attachments/assets/7361e0a8-8f88-4731-879d-07b8a6458fdb" />

En la siguiente imagen se muestra el diseño de arquitectura híbrida de la solución del TFM.
<img width="857" height="656" alt="disenoTFM" src="https://github.com/user-attachments/assets/46da0fe2-e79f-4aba-91ad-9942a30f322a" />

Para ejecutar los archivos de python (*.py) localmente en el equipo se deben seguir las siguientes instrucciones:
- Instalar la versión de Python 3.11 desde el instalador disponible https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe.
  Utilizar como carpeta raiz c:\Python311
- Copiar todos los archivos de la raiz de este repositorio en la carpeta c:\agente_mt5
- Crear un ambiente virtual ejecutando:
  cd C:\agente_mt5
  C:\Python311\python.exe -m venv venv

- Activar el entorno ejecutando:
  venv\Scripts\activate.bat

- Se debe mostrar un prompt similar a:
  (venv) C:\agente_mt5>

- Instalar las dependencias:
  pip install --upgrade pip
  pip install MetaTrader5==5.0.45 onnxruntime==1.19.2 numpy==1.26.4 pandas==2.2.3 pydantic==2.9.2


Módulo 1 — config.py (Configuración centralizada)
¿Qué hace este módulo?
Es el único archivo que debes editar para controlar todo el comportamiento del agente. Centraliza símbolo, timeframe, parámetros del modelo, SL/TP, y los valores del scaler generados en Colab.

- Verificar el archivo config.py
  python -c "from config import cfg; print('OK'); print('Symbol:', cfg.symbol); print('Modelo:', cfg.modelo_path)"


Módulo 2 — features.py
¿Qué hace este módulo?
Descarga las velas desde MT5, calcula exactamente las 16 features que usó el modelo durante el entrenamiento, detecta si hay una señal válida, y normaliza el resultado con los parámetros del scaler.

- Verificar que el archivo features.py no contenga errores de sintaxis
  python -c "import features; print('features.py OK')"

- Verificar que las features coinciden con el modelo
  echo from config import cfg > test_mod2.py
  echo print('Features esperadas por el modelo:', len(cfg.scaler_mean)) >> test_mod2.py
  echo print('Features que calcula features.py : 16') >> test_mod2.py
  echo print('Coinciden:', len(cfg.scaler_mean) == 16) >> test_mod2.py

  python test_mod2.py


Módulo 3 — riesgo.py (Gestión de riesgo)
¿Qué hace este módulo?
Antes de enviar cualquier orden al broker, el agente consulta este módulo. Verifica cuatro condiciones en orden. Si alguna falla, el agente no opera y registra el motivo. Nunca se salta esta verificación.

- Verificar que el archivo riesgo.py no contenga errores de sintaxis
  python -c "import riesgo; print('riesgo.py OK')"

- Prueba de las funciones con MT5 abierto
  echo import MetaTrader5 as mt5 > test_mod3.py
  echo from riesgo import puede_operar, resumen_cuenta >> test_mod3.py
  echo mt5.initialize() >> test_mod3.py
  echo ok, motivo = puede_operar() >> test_mod3.py
  echo print('Puede operar:', ok) >> test_mod3.py
  echo print('Motivo:', motivo if not ok else 'Sin restricciones') >> test_mod3.py
  echo resumen = resumen_cuenta() >> test_mod3.py
  echo print('Balance:', resumen.get('balance', 'N/A')) >> test_mod3.py
  echo print('Equity:', resumen.get('equity', 'N/A')) >> test_mod3.py
  echo print('Drawdown:', resumen.get('drawdown', 'N/A'), '%') >> test_mod3.py
  echo mt5.shutdown() >> test_mod3.py

  python test_mod3.py


Módulo 4 — ejecutor.py (Envío de órdenes)
¿Qué hace este módulo?
Es el único punto del agente que envía órdenes al broker. Calcula el precio de entrada, SL y TP, construye la orden, la envía a MT5, verifica el resultado, reintenta si falla, y registra cada operación en el CSV.

- Verificar que el archivo ejecutor.py no contenga errores de sintaxis
  python -c "import ejecutor; print('ejecutor.py OK')"

- Prueba de las funciones
  echo import MetaTrader5 as mt5 > test_mod4.py
  echo from riesgo import calcular_precios_long >> test_mod4.py
  echo mt5.initialize() >> test_mod4.py
  echo entry, sl, tp = calcular_precios_long() >> test_mod4.py
  echo print('Entry:', entry) >> test_mod4.py
  echo print('SL   :', sl) >> test_mod4.py
  echo print('TP   :', tp) >> test_mod4.py
  echo print('SL valido:', sl ^< entry) >> test_mod4.py
  echo print('TP valido:', tp ^> entry) >> test_mod4.py
  echo mt5.shutdown() >> test_mod4.py

  python test_mod4.py


Módulo 5 — agente.py (Loop principal)
¿Qué hace este módulo?
Es el cerebro del agente. Conecta todos los módulos anteriores en un ciclo continuo: descarga datos → calcula features → verifica señal → consulta el modelo → verifica riesgo → ejecuta la orden → espera al siguiente ciclo. Se ejecuta indefinidamente hasta que presionas Ctrl+C.

- Verificar que el archivo agente.py no contenga errores de sintaxis
  python -c "import agente; print('agente.py OK')"

- Prueba de ejecución del agente
  python agente.py
