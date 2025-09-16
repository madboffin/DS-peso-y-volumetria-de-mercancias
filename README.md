# DS-peso-y-volumetria-de-mercancias


## Posibles estrategias para la Estimación de Volumen y Peso de Mercancías

La precisión en el cálculo de volumen y peso de las mercancías es fundamental para la logística, el almacenamiento y la planificación de envíos. Este repositorio explora y implementa diversas metodologías para abordar este desafío, desde enfoques basados en reglas hasta modelos avanzados de Machine Learning.

A continuación, se detallan las principales estrategias consideradas como **posibles alternativas para calcular volúmenes y pesos de mercancías**:

### 1. Extracción y Normalización por Heurísticas
*   **Descripción:** Identificar, extraer y normalizar valores explícitos de peso y volumen directamente de las descripciones y nombres de los productos. Esto incluye patrones como "500 g", "1 L" o "pack 6 x 330 ml".
*   **Beneficios:** Eficiente como estrategia de base y proporciona una cobertura inmediata para una gran cantidad de productos con información explícita.

### 2. Estimación Basada en Densidad por Categoría
*   **Descripción:** Cuando se disponen de dimensiones, se calcula el peso multiplicando el volumen por una densidad promedio estimada para la categoría del producto (ej. líquidos ≈1 g/cm³, sólidos varían según el tipo).
*   **Beneficios:** Muy útil para inferir pesos cuando no están explícitamente registrados, aprovechando el conocimiento sobre la composición material de las categorías.

### 3. Modelos de Regresión Supervisada
*   **Descripción:** Desarrollo y entrenamiento de modelos de Machine Learning (como RandomForest, Gradient Boosting Machines, XGBoost o LightGBM) para predecir el peso y/o las dimensiones. Estos modelos utilizan un conjunto de *features* disponibles, tales como la categoría del producto, el texto del nombre, volumen declarado, unidades, dimensiones parciales, marca, tamaño del empaque, etc.
*   **Beneficios:** Permite predicciones más precisas y adaptadas a la complejidad de los datos, capturando relaciones no lineales entre las características.

### 4. Inferencia por Similitud (NLP y Nearest-Neighbor / Embeddings)
*   **Descripción:** Para productos con datos incompletos o totalmente faltantes, se emplean técnicas de Procesamiento de Lenguaje Natural (NLP) y embeddings para mapear el producto nuevo a otros productos textualmente similares en la base de datos. Se utilizan los valores promedio o la mediana de peso/volumen de sus vecinos más cercanos.
*   **Beneficios:** Ideal para la imputación de datos, la gestión de productos nuevos y la inferencia en escenarios de escasez de información directa.

### 5. Enfoque Híbrido y Reglas de Negocio
*   **Descripción:** Una estrategia combinada que prioriza las heurísticas para casos sencillos y directos, recurre a modelos de Machine Learning cuando las heurísticas son insuficientes, y aplica reglas de negocio específicas (ej. límites mínimos/máximos por categoría, ajustes predefinidos) para el post-procesamiento y la validación final.
*   **Beneficios:** Garantiza una solución robusta, flexible y alineada con los requisitos operacionales y las particularidades del negocio, optimizando la precisión y la eficiencia.

### 6. Uso de APIs de modelos LLM
*   **Descripción:** Utilización de modelos de lenguaje grande (LLM) avanzados, como GPT o Gemini, a través de sus APIs. Se ingresan las descripciones de los productos, nombres, o cualquier texto relevante como prompts. El LLM procesa esta información y genera una estimación del peso y volumen, a partir del vasto conocimiento que ha adquirido sobre productos, unidades de medida y el mundo real en su entrenamiento.


*   **Beneficios:** incluso con descripciones ambiguas, incompletas o en lenguaje natural. Los LLM pueden comprender el contexto, el significado de las palabras y realizar razonamiento complejo, lo que los hace ideales para manejar la heterogeneidad y falta de estructura de los datos textuales en logística.



# Como ejecutar el proyecto

Se espera la ubicacion del archivo de entrada, y la del archivo de salida.
## Para hacer predicciones:

`docker-compose up --build`

`docker-compose run --rm product-predictor python cli/main.py predict --input /app/data/raw/new_products.csv --output /app/data/processed/predictions.csv`

## Modo interactivo:
`docker-compose run --rm product-predictor bash`

Para correr el proyecto sin usar docker:

`python -m cli.main predict -i /data/raw/market_products.csv -o /data/processed/predictions.csv`
