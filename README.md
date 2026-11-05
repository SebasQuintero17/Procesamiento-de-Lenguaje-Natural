# Proyecto 3 — Procesamiento de Lenguaje Natural
### Clasificación de emociones en texto con *fine-tuning* de modelos tipo BERT

**Curso:** Machine Learning — Universidad de Medellín
**Estudiante:** Sebastián Quintero Granada

## Descripción
Fine-tuning de modelos Transformer preentrenados (**DistilBERT** y **BERT-base**) para clasificar la emoción expresada en tweets en inglés en 6 clases: *sadness, joy, love, anger, fear, surprise*.

| | |
|---|---|
| **Dataset** | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face) — 16.000 train / 2.000 validation / 2.000 test |
| **Tipo de problema** | Sequence classification multiclase (6 clases) |
| **Modelos** | `distilbert-base-uncased`, `bert-base-uncased` |
| **Hiperparámetros explorados** | learning rate {2e-5, 3e-5, 5e-5} × batch size {16, 32}; épocas {3, 5}; arquitectura |
| **Métrica de selección** | F1 macro en validación (clases desbalanceadas) |

## Contenido del notebook
1. Configuración del entorno
2. Descripción de la base de datos (dominio, utilidad, entrada, salida, tipo de problema)
3. Análisis exploratorio (ejemplos por clase, distribución, longitudes, vocabulario)
4. Tokenización
5. Fine-tuning y malla de hiperparámetros
6. Comparación DistilBERT vs. BERT-base
7. Mejor modelo: evaluación en test, matriz de confusión y análisis de errores
8. Inferencia con frases nuevas
9. Conclusiones

## Cómo ejecutarlo
1. Abrir `Proyecto_NLP_Emociones_BERT.ipynb` en Google Colab.
2. `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)`.
3. `Entorno de ejecución → Ejecutar todas`. Tiempo aproximado: 30–40 min.

El dataset se descarga directamente desde Hugging Face dentro del notebook; no es necesario subirlo al repositorio.

## Tecnologías
Python · PyTorch · Hugging Face `transformers`, `datasets` · scikit-learn · pandas · matplotlib / seaborn
