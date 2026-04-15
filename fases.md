---

# ADAMO ID — Blueprint Técnico y Plan de Implementación
## API de Verificación de Autenticidad de Documentos de Identidad

---

## Resumen Ejecutivo

La solución propuesta consiste en un pipeline de 5 modelos de inteligencia artificial especializados, desplegados como un único endpoint en RunPod con GPU, capaz de verificar la autenticidad de imágenes de documentos de identidad con una precisión superior al 95% por filtro.

El costo operativo estimado es de $26 a $108 USD al mes dependiendo del volumen de verificaciones (10,000 a 50,000 mensuales), lo que representa un ahorro de entre 50 y 200 veces en comparación con APIs comerciales de KYC como Onfido o Mitek, que cobran entre $0.10 y $0.50 por verificación.

Toda la solución se construye sobre modelos open-source y código propio, sin dependencia de servicios de terceros. El cliente mantiene control total sobre los modelos, la infraestructura y los costos.

---

## Los 5 Filtros de Autenticidad

Cada filtro responde una pregunta específica y retorna:

```json
{
  "answer": "yes / no",
  "percentageOfConfidence": 94.5
}
```

---

### Filtro 1 — Detección de Captura de Pantalla

**Pregunta:** ¿La foto del documento fue tomada desde una pantalla (monitor, laptop, celular)?

**Modelo principal:** CMA — Chromaticity Map Adapter con ViT-B/16 (CVPR 2024)

Este modelo detecta los artefactos físicos que ocurren cuando una cámara fotografía una pantalla: desalineación de subpíxeles, patrones de interferencia (efecto moiré), cambios de cromaticidad y distorsión de gamma. Estos artefactos son invisibles al ojo humano pero detectables con alta precisión.

- AUC: 0.899 en benchmark ROD
- AUC bajo compresión JPEG (QF=70): 0.869
- Tiempo de inferencia: 8 a 12ms en RTX 4090
- VRAM requerida: 1.5GB

**Alternativa complementaria:** EfficientNet-B4 fine-tuneado sobre datos de recaptura, con inferencia de 3 a 5ms y precisión cercana al 90%.

---

### Filtro 2 — Detección de Documento Impreso

**Pregunta:** ¿El documento fue impreso en papel y luego fotografiado?

**Modelo principal:** EfficientNet-B4 con augmentación FHAG (Frequency-domain Halftoning Augmentation with Band-of-Interest Localization)

Los documentos impresos contienen patrones de medios tonos del proceso de impresión (tóner o tinta) que no existen en documentos originales laminados. El modelo analiza el dominio de frecuencias para detectar estos patrones espectrales específicos que delatan una reproducción impresa.

- Reducción de tasa de error: 25% frente a métodos estándar
- Tiempo de inferencia: 3 a 5ms en RTX 4090
- VRAM requerida: 1GB

Nota técnica: los filtros 1 y 2 pueden compartir el backbone de EfficientNet-B4 con cabezas de clasificación separadas, reduciendo el consumo total de VRAM a aproximadamente 1.2GB sin afectar la precisión.

---

### Filtro 3 — Detección de Elementos Superpuestos

**Pregunta:** ¿El documento tiene elementos superpuestos como stickers, datos alterados o una foto de otro rostro pegada encima?

**Modelo principal:** TruFor (CVPR 2023)

TruFor combina análisis RGB con Noiseprint++, que detecta la huella digital de ruido de la cámara a nivel de píxel. Cualquier elemento pegado, editado o alterado interrumpe esta huella y genera una anomalía detectable con precisión quirúrgica.

Produce tres salidas:
- Mapa de calor a nivel de píxel indicando exactamente dónde ocurrió la manipulación
- Puntuación de integridad general de la imagen
- Mapa de confiabilidad para reducir falsos positivos

**Modelo secundario:** HiFi-IFDL (licencia MIT, apto para uso comercial). Clasifica el tipo de falsificación: imagen completamente sintética, manipulación parcial, o método específico.

**Modelo especializado en documentos:** DocTamper (CVPR 2023), entrenado con 170,000 imágenes de documentos. Mejora la detección de texto alterado entre 9 y 26% frente a detectores genéricos.

- Tiempo de inferencia combinado: 50 a 100ms en RTX 4090
- VRAM requerida: 4 a 6GB

---

### Filtro 4 — Detección de Alteración por IA

**Pregunta:** ¿La imagen del documento fue generada o modificada por inteligencia artificial?

**Modelo principal:** UnivFD — Universal Fake Detector (CVPR 2023)

Utiliza el espacio de características de CLIP ViT-L/14 con una sola capa de clasificación lineal. Generaliza a más de 19 modelos generativos incluyendo StyleGAN, Stable Diffusion, DALL-E, Midjourney y Glide. Supera al estado del arte anterior en +15.07 mAP en modelos de difusión no vistos durante entrenamiento.

- Tiempo de inferencia: 5 a 10ms en RTX 4090
- VRAM requerida: 3GB

**Modelo secundario:** DIRE — Diffusion Reconstruction Error (ICCV 2023). Detecta imágenes generadas por modelos de difusión con 99.9% de precisión. Por su mayor latencia (2 a 5 segundos), se activa únicamente cuando UnivFD retorna un puntaje ambiguo entre 30% y 70% de confianza.

**Para detección de rostros intercambiados:** SBI — Self-Blended Images (CVPR 2022), que detecta artefactos de fusión en los bordes del rostro pegado sobre el documento original.

---

### Filtro 5 — Detección de Vida Real (Liveness)

**Pregunta:** ¿La foto fue tomada en tiempo real por un usuario real sosteniendo el documento?

Este filtro utiliza un sistema de puntuación compuesta basado en 6 señales independientes:

| Señal | Peso | Método |
|---|---|---|
| Resultado de Filtro 1 (no es pantalla) | 25% | Modelo CMA |
| Resultado de Filtro 2 (no es impresión) | 20% | EfficientNet-B4 |
| Anti-spoofing del rostro en el documento | 20% | MiniFASNetV2 — 98% precisión, 600KB |
| Análisis de perspectiva y bordes | 15% | OpenCV — un documento sostenido tiene distorsión natural |
| Detección de manos y dedos en los bordes | 10% | MediaPipe Hands (CPU, sin costo GPU) |
| Análisis de metadatos EXIF | 10% | Pillow — modelo de cámara, timestamp, doble compresión |

La puntuación final compuesta entrega una precisión estimada del 92 al 96% sobre los ataques más comunes: captura de pantalla, copia impresa y sustitución de retrato.

---

## Arquitectura de Deployment en RunPod

### Por qué RunPod Serverless Flex

Con un volumen de 10,000 a 50,000 verificaciones mensuales, la GPU permanece inactiva entre el 93% y el 99% del tiempo. Un pod siempre encendido cuesta $158 USD al mes sin importar el uso. El modo serverless flex factura únicamente los segundos de cómputo activo, reduciendo el costo entre 10 y 20 veces.

**GPU recomendada:** A4000 o A4500 de 16GB VRAM a $0.000160 por segundo. Los 5 modelos combinados consumen entre 8 y 10GB de VRAM, con margen suficiente en la GPU de 16GB.

### Estructura del contenedor

Un único contenedor Docker con todos los modelos cargados al inicio. Esta es la arquitectura correcta porque:

- Todos los modelos caben en una sola GPU de 16GB
- La inferencia secuencial interna (3.5s promedio) es más rápida que 5 llamadas de red a contenedores separados
- Garantiza que los 5 filtros se ejecuten sobre la misma imagen de forma atómica
- Con RunPod FlashBoot, los cold starts subsiguientes se reducen a menos de 500ms

### Costo mensual estimado

| Componente | 10,000 verif/mes | 50,000 verif/mes |
|---|---|---|
| Cómputo GPU (serverless flex) | $25.12 | $106.40 |
| Disco del contenedor (10GB) | $1.00 | $1.00 |
| Volumen de red (5GB) | $0.35 | $0.35 |
| **Total estimado** | **$26/mes** | **$108/mes** |

Costo por verificación: $0.0026 a escala de 10,000 y $0.0022 a escala de 50,000.

---

## Formato de la API

**Endpoint principal:**

```
POST /verify
```

**Request:**

```json
{
  "image": "base64_encoded_string",
  "options": {
    "return_heatmaps": false,
    "confidence_threshold": 0.5
  }
}
```

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "processing_time_ms": 3245,
  "verdict": {
    "is_authentic": true,
    "overall_confidence": 94.2,
    "risk_level": "low"
  },
  "filters": {
    "screen_capture": { "answer": "no", "percentageOfConfidence": 96.3 },
    "printed_paper": { "answer": "no", "percentageOfConfidence": 93.1 },
    "superimposed_elements": { "answer": "no", "percentageOfConfidence": 97.8 },
    "ai_altered": { "answer": "no", "percentageOfConfidence": 95.5 },
    "liveness": { "answer": "yes", "percentageOfConfidence": 91.2 }
  }
}
```

---

## Stack Tecnológico Completo

**Lenguaje:** Python 3.11

**Framework de inferencia:** PyTorch 2.x con CUDA 12.x

**API:** FastAPI con Pydantic v2 para validación de esquemas

**Deployment:** RunPod Serverless con handler nativo (runpod SDK)

**Contenedor:** Docker — imagen base NVIDIA CUDA 12.1, pesos de modelos incluidos (~8GB total)

| Filtro | Modelo | Licencia |
|---|---|---|
| Screen capture | CMA ViT-B/16 | Académica |
| Print detection | EfficientNet-B4 + FHAG | Apache 2.0 |
| Forgery localization | TruFor / HiFi-IFDL | No comercial / MIT |
| AI detection | UnivFD + CLIP ViT-L/14 | MIT |
| Liveness composite | MiniFASNetV2 + MediaPipe + OpenCV | Apache 2.0 |

Nota sobre licencias: TruFor tiene licencia no comercial. Para deployment completamente comercial se reemplaza por HiFi-IFDL (MIT) como detector principal de falsificaciones, con una leve reducción de precisión aceptable en producción.

---

## Datasets para Entrenamiento y Fine-Tuning

| Dataset | Descripción | Licencia |
|---|---|---|
| SIDTD | Documentos falsificados por crop-and-replace e inpainting, 10 nacionalidades europeas | CC BY-SA 2.5 |
| DLC-2021 | 1,424 clips de video con pantalla, copia y documentos genuinos | Libre |
| IDNet (2024) | 837,060 imágenes sintéticas de 20 tipos de documentos con face morphing y alteración de texto | Zenodo |
| MIDV-2020 | 72,409 imágenes anotadas de 1,000 IDs únicos | CC BY-SA 2.5 |
| DocTamper | 170,000 imágenes de documentos con texto alterado | No comercial |

---

## Plan de Implementación por Fases

---

### Fase 1 — Fundación e Infraestructura

**Objetivo:** Tener la infraestructura base operativa y los dos primeros filtros funcionando end-to-end en RunPod.

**Actividades:**

Configuración del entorno de desarrollo con PyTorch 2.x, CUDA 12.x y las dependencias base. Creación del proyecto en RunPod: configuración del endpoint serverless, network volume y política de escalado. Construcción del Dockerfile base con imagen NVIDIA CUDA 12.1 e instalación de dependencias.

Integración del Filtro 1 (Screen Capture): descarga y carga del modelo CMA ViT-B/16, preprocesamiento de imagen (resize, normalización de cromaticidad), función de inferencia con salida de confianza normalizada entre 0 y 100.

Integración del Filtro 2 (Print Detection): descarga del backbone EfficientNet-B4 preentrenado en ImageNet, implementación del preprocesamiento de dominio de frecuencias (FFT para extracción de banda de medios tonos), cabeza de clasificación de 2 clases, función de inferencia.

Implementación del handler RunPod con los primeros 2 filtros. Endpoint FastAPI `/verify` retornando el JSON parcial de los 2 filtros. Primera ronda de pruebas con imágenes del dataset DLC-2021.

**Entregable:** Endpoint en RunPod operativo retornando resultados de Filtros 1 y 2 con precisión inicial validada sobre dataset de prueba.

---

### Fase 2 — Detección de Falsificaciones y Alteración por IA

**Objetivo:** Integrar los filtros de mayor complejidad técnica: detección de elementos superpuestos y detección de alteración por IA.

**Actividades:**

Integración del Filtro 3 (Superimposed Elements): descarga e integración de TruFor con sus pesos preentrenados, implementación del pipeline de ensemble (TruFor + HiFi-IFDL), función de agregación de scores que combina el integrity score de TruFor con la clasificación de HiFi-IFDL, integración de DocTamper para refuerzo en detección de texto alterado, normalización del score compuesto a porcentaje de confianza.

Integración del Filtro 4 (AI-Altered Detection): integración de UnivFD con backbone CLIP ViT-L/14, implementación del pipeline condicional: UnivFD como detector primario rápido, activación de DIRE como detector secundario cuando la confianza de UnivFD cae entre 30% y 70%, integración de SBI para detección de face swap en la región del retrato del documento.

Actualización del handler RunPod con los 4 filtros. Pruebas de integración completas. Validación de VRAM: los 4 modelos deben mantenerse bajo los 16GB de la GPU. Medición de latencia end-to-end con los 4 filtros activos.

**Entregable:** Endpoint operativo con 4 filtros funcionando, latencia end-to-end medida y documentada, reporte de precisión por filtro sobre dataset de validación.

---

### Fase 3 — Filtro de Liveness y Sistema de Scoring Final

**Objetivo:** Completar el quinto filtro e integrar el sistema de veredicto final unificado.

**Actividades:**

Integración del Filtro 5 (Liveness Composite): integración de MiniFASNetV2 (ONNX) para anti-spoofing del retrato del documento, implementación del detector de perspectiva con OpenCV (verificación de distorsión natural de un documento sostenido), integración de MediaPipe Hands para detección de dedos en los bordes del documento, implementación del analizador de metadatos EXIF con Pillow (doble compresión, modelo de cámara, timestamp), implementación de la función de scoring ponderado que combina las 6 señales del filtro.

Implementación del sistema de veredicto final: función `aggregate_verdict` que combina los 5 filtros con pesos configurables, cálculo del `overall_confidence` y `risk_level` (low / medium / high), umbral configurable por el cliente vía `confidence_threshold`.

Actualización del response completo al formato final definido. Pruebas end-to-end con los 5 filtros activos. Medición de latencia total (objetivo: bajo 4 segundos por verificación). Pruebas de carga con solicitudes concurrentes.

**Entregable:** Endpoint completamente funcional con los 5 filtros y el veredicto final. JSON de respuesta en el formato exacto solicitado. Reporte de latencia y precisión por filtro.

---

### Fase 4 — Fine-Tuning, Hardening y Entrega Final

**Objetivo:** Optimizar la precisión mediante fine-tuning sobre datasets de documentos reales, endurecer el sistema para producción KYC y entregar la solución completa.

**Actividades:**

Fine-tuning de los modelos de los filtros 1 y 2 sobre SIDTD + DLC-2021: preparación del dataset (split 80/20 train/val), entrenamiento con backbone congelado por 10 épocas (lr=1e-3), luego descongelado parcial por 30 a 50 épocas (lr=1e-5) con AdamW y cosine annealing, evaluación de AUC, precisión y recall sobre conjunto de validación.

Hardening para producción KYC: validación de inputs (dimensiones máximas 4096x4096, tamaño máximo 10MB, formatos permitidos JPEG, PNG, WEBP), manejo robusto de errores con respuestas estructuradas en lugar de crashes, logging de cada solicitud con request_id, filtro activado, score y latencia, configuración de escalado RunPod (3 a 5 workers máximo, Queue Delay threshold de 4 segundos).

Documentación técnica de la API: especificación OpenAPI completa, guía de integración para el equipo de KYC del cliente, descripción de cada filtro y su interpretación, guía de configuración de umbrales según el nivel de tolerancia al riesgo del cliente.

Entrega del repositorio: código fuente completo, Dockerfile listo para producción, scripts de fine-tuning reproducibles, configuración de RunPod exportable, suite de pruebas automatizadas.

**Entregable:** Sistema en producción con precisión superior al 95% por filtro validada sobre datos reales. Repositorio completo con documentación. Configuración RunPod lista para el cliente. Guía de mantenimiento y actualización de modelos.

---

## Resumen de Fases y Entregables

| Fase | Objetivo principal | Filtros completados | Entregable clave |
|---|---|---|---|
| Fase 1 | Infraestructura base | Filtros 1 y 2 | Endpoint RunPod funcional con 2 filtros |
| Fase 2 | Detección de falsificaciones e IA | Filtros 3 y 4 | Endpoint con 4 filtros, reporte de precisión |
| Fase 3 | Liveness y scoring final | Filtro 5 + veredicto | Sistema completo con respuesta final |
| Fase 4 | Fine-tuning y producción | Optimización de todos | Repositorio completo + sistema en producción |

---

## Conclusión

Esta arquitectura entrega verificación de autenticidad de documentos en cinco capas a una fracción del costo de APIs comerciales. La clave está en que ningún modelo único cubre todos los vectores de ataque: la detección de pantalla requiere análisis de cromaticidad, la detección de impresión necesita análisis de medios tonos, la detección de falsificaciones exige análisis de anomalías a nivel de píxel, y la detección de alteración por IA aprovecha el espacio semántico de CLIP. El scoring compuesto de liveness logra precisión robusta precisamente porque fusiona señales ortogonales e independientes.

El cliente obtiene una solución con control total sobre el código, los modelos y la infraestructura, sin dependencia de terceros, con un costo operativo predecible y escalable, y con la flexibilidad de ajustar umbrales y pesos según la política de riesgo de su aplicación KYC.