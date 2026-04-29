# ADAMO ID — Documentación Técnica de la API

**Versión**: 0.2.0  
**Motor**: Gemini Vision (Google AI) + análisis forense local  
**Protocolo**: REST / HTTP  
**Formato**: JSON  

---

## Tabla de contenidos

1. [Descripción general](#1-descripción-general)  
2. [Arquitectura y filtros](#2-arquitectura-y-filtros)  
3. [Instalación y configuración](#3-instalación-y-configuración)  
4. [Endpoints de la API](#4-endpoints-de-la-api)  
5. [Estructura del request y response](#5-estructura-del-request-y-response)  
6. [Ejemplos de uso](#6-ejemplos-de-uso)  
7. [Script CLI para pruebas](#7-script-cli-para-pruebas)  
8. [Códigos de error](#8-códigos-de-error)  
9. [Despliegue con Docker](#9-despliegue-con-docker)  
10. [Preguntas frecuentes](#10-preguntas-frecuentes)  

---

## 1. Descripción general

**ADAMO ID** es una API de verificación de autenticidad de documentos de identidad (cédulas colombianas, pasaportes y otros documentos oficiales). Analiza una fotografía del documento e indica, con grado de confianza, si el documento es auténtico o si fue alterado de alguna manera.

### ¿Qué detecta?

| # | Tipo de fraude | Descripción |
|---|---|---|
| 1 | **Captura de pantalla** | El documento se fotografió desde la pantalla de un computador, celular o televisor |
| 2 | **Impresión en papel** | El documento fue impreso en papel (fotocopia o impresión a color) y re-fotografiado |
| 3 | **Elementos superpuestos** | El documento tiene stickers, parches, fotos pegadas o datos adulterados encima |
| 4 | **Generado por IA** | El documento fue creado o modificado con inteligencia artificial |
| 5 | **Autenticidad consolidada** | Score global que resume los 4 filtros anteriores (el cliente lo llama "el filtro 5") |

### ¿Cómo funciona?

```
Imagen (base64)
      │
      ▼
┌─────────────────────────────────────────┐
│  Análisis forense local (numpy/PIL)     │  ← ~50ms en CPU, sin red
│  · Halftone FFT score                   │
│  · Saturación de color (p95)            │
│  · Puntos especulares (laminado)        │
│  · Varianza de bordes (Laplaciano)      │
└─────────────┬───────────────────────────┘
              │ métricas numéricas
              ▼
┌─────────────────────────────────────────┐
│  Gemini Vision API (1 sola llamada)     │  ← ~7-15s por imagen
│  · Evalúa los 4 tipos de fraude        │
│  · Usa las métricas forenses como      │
│    contexto adicional                  │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│  Pipeline de agregación                 │
│  · Filtros 1-4: resultado de Gemini    │
│  · Filtro 5 (liveness): derivado de   │
│    los anteriores (score consolidado)  │
│  · Veredicto final con riesgo         │
└─────────────────────────────────────────┘
```

---

## 2. Arquitectura y filtros

### Los 5 filtros explicados

#### Filtro 1 — `screen_capture` (Captura de pantalla)
Detecta cuando el documento no está siendo fotografiado directamente, sino que se muestra en una pantalla y esa pantalla es fotografiada.

**Señales que detecta:**
- Cursor del mouse visible
- Bordes/bisel del dispositivo alrededor del documento
- Moiré (patrón de interferencia entre píxeles de la pantalla)
- El documento más brillante que el entorno (auto-iluminación de pantalla)
- Barras de estado o notificaciones del sistema operativo

#### Filtro 2 — `printed_paper` (Impresión en papel)
Detecta cuando la cédula fue impresa en papel (impresora de tinta o láser) y luego fotografiada.

**Señales que detecta:**
- Colores desaturados, rosados o amarillentos (típico de fotocopias)
- Retrato blando/difuso sin el grabado láser característico de la cédula real
- Borde del documento completamente plano (sin espesor de 0.8mm del policarbonato)
- Fondo guilloche borroso o desvanecido
- Patrones de puntos CMYK visibles al ampliar
- Documento sobre tela, sábana, cuaderno o papel arrugado
- Registro de color desalineado (bordes de color alrededor de texto)

#### Filtro 3 — `superimposed_elements` (Elementos superpuestos)
Detecta cuando se colocaron elementos encima del documento original para alterar su información.

**Señales que detecta:**
- Stickers o parches opacos sobre campos de datos
- Foto pegada encima del retrato original (bordes de corte visibles)
- Patrón de seguridad que se interrumpe en los bordes de la foto
- Datos de texto que no pertenecen al diseño original del documento
- Inconsistencias de datos (fecha/registrador/NUIP que no corresponden)

#### Filtro 4 — `ai_altered` (Generado por IA)
Detecta documentos creados o modificados con herramientas de inteligencia artificial generativa.

**Señales que detecta:**
- Rasgos faciales que se "derriten" o difuminan en los bordes
- Texto con tipografía que no corresponde al diseño oficial de la cédula
- Ojos con color irreal (brillante, uniforme, sobrenatural)
- Cabello con textura sintética o bordes disueltos
- Simetría facial artificial

> ⚠️ **Importante:** El filtro de IA NO marca como sospechoso un retrato joven, con piel suave, ojos cálidos o buena iluminación. Solo se activa ante artefactos concretos de generación artificial.

#### Filtro 5 — `liveness` (Autenticidad consolidada)
Es el **score global de autenticidad** del documento. No es un detector independiente: se calcula derivando el resultado de los filtros 1 al 4.

- Si **ningún** filtro detectó fraude → `answer: "no"`, confianza = el mínimo de los 4 filtros (la garantía más débil)
- Si **algún** filtro detectó fraude → `answer: "yes"`, confianza = el máximo de los detectores que dispararon

Este filtro responde directamente a la pregunta del cliente: *"¿En qué grado de confianza es auténtica?"*

---

## 3. Instalación y configuración

### Requisitos previos

- Python 3.11+
- Una API Key de Google Gemini (`GEMINI_API_KEY`)
- Docker (opcional, recomendado para producción)

### Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
# Obligatorio
GEMINI_API_KEY=AIza...tu_clave_aqui...

# Opcional (valores por defecto mostrados)
LOG_LEVEL=INFO
MAX_IMAGE_BYTES=10485760   # 10 MB
MAX_DIMENSION=4096          # px
```

### Instalación local (desarrollo)

```bash
# 1. Clonar el repositorio
git clone https://github.com/DTv2-1/id-recognition.git
cd id-recognition

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar credenciales
cp .env.example .env
# Editar .env y agregar GEMINI_API_KEY

# 5. Levantar la API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La API quedará disponible en: `http://localhost:8000`  
Documentación interactiva (Swagger): `http://localhost:8000/docs`

---

## 4. Endpoints de la API

### `GET /health`

Verifica que el servicio esté activo.

**Response:**
```json
{
  "status": "ok",
  "engine": "gemini"
}
```

| Campo | Valores | Descripción |
|---|---|---|
| `status` | `"ok"` | Servicio activo |
| `engine` | `"gemini"` / `"local"` | Motor activo (Gemini si hay API key, local como fallback) |

---

### `POST /verify`

Analiza una imagen de documento de identidad y retorna el veredicto completo.

**Headers:**
```
Content-Type: application/json
```

**Body:** Ver sección [5. Estructura del request](#5-estructura-del-request-y-response)

**Response:** Ver sección [5. Estructura del response](#response)

---

## 5. Estructura del request y response

### Request

```json
{
  "image": "<base64_de_la_imagen>",
  "options": {
    "confidence_threshold": 0.5,
    "return_heatmaps": false
  }
}
```

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `image` | `string` | ✅ Sí | Imagen en Base64. Formatos: **JPEG, PNG, WEBP**. Tamaño máximo: **10 MB**. Dimensiones máximas: **4096×4096 px** |
| `options.confidence_threshold` | `float` | No | Umbral mínimo de confianza para considerar el documento auténtico. Rango: 0.0–1.0. Defecto: `0.5` |
| `options.return_heatmaps` | `bool` | No | Reservado para versiones futuras. Por ahora siempre `false` |

> **Nota sobre el Base64:** La API acepta tanto Base64 puro como Data URLs con prefijo (`data:image/jpeg;base64,...`).

### Response

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "processing_time_ms": 8432,
  "verdict": {
    "is_authentic": true,
    "overall_confidence": 95.0,
    "risk_level": "low"
  },
  "filters": {
    "screen_capture": {
      "answer": "no",
      "percentageOfConfidence": 95.0
    },
    "printed_paper": {
      "answer": "no",
      "percentageOfConfidence": 95.0
    },
    "superimposed_elements": {
      "answer": "no",
      "percentageOfConfidence": 95.0
    },
    "ai_altered": {
      "answer": "no",
      "percentageOfConfidence": 95.0
    },
    "liveness": {
      "answer": "no",
      "percentageOfConfidence": 95.0
    }
  }
}
```

#### Campo `verdict`

| Campo | Tipo | Descripción |
|---|---|---|
| `is_authentic` | `bool` | `true` = documento auténtico. `false` = fraude detectado |
| `overall_confidence` | `float` (0–100) | Grado de confianza de autenticidad en porcentaje |
| `risk_level` | `string` | Nivel de riesgo: `"low"` / `"medium"` / `"high"` |

**Cálculo de `risk_level`:**

| `overall_confidence` | `risk_level` |
|---|---|
| ≥ 80% | `"low"` — bajo riesgo, documento probablemente auténtico |
| 60% – 79% | `"medium"` — revisar manualmente |
| < 60% | `"high"` — alto riesgo de fraude |

#### Campo `filters`

Cada uno de los 5 filtros retorna:

| Campo | Tipo | Descripción |
|---|---|---|
| `answer` | `"yes"` / `"no"` | `"yes"` = se detectó ese tipo de fraude. `"no"` = no se detectó |
| `percentageOfConfidence` | `float` (0–100) | Confianza del filtro en su respuesta |

> **Regla de lectura de filtros:**
> - `answer: "no", percentageOfConfidence: 95` → el filtro está 95% seguro de que **no hay** ese tipo de fraude
> - `answer: "yes", percentageOfConfidence: 98` → el filtro está 98% seguro de que **sí hay** ese tipo de fraude

#### Casos de respuesta completos

**Documento auténtico:**
```json
{
  "verdict": { "is_authentic": true, "overall_confidence": 95.0, "risk_level": "low" },
  "filters": {
    "screen_capture":        { "answer": "no",  "percentageOfConfidence": 95.0 },
    "printed_paper":         { "answer": "no",  "percentageOfConfidence": 95.0 },
    "superimposed_elements": { "answer": "no",  "percentageOfConfidence": 95.0 },
    "ai_altered":            { "answer": "no",  "percentageOfConfidence": 95.0 },
    "liveness":              { "answer": "no",  "percentageOfConfidence": 95.0 }
  }
}
```

**Cédula impresa en papel:**
```json
{
  "verdict": { "is_authentic": false, "overall_confidence": 2.0, "risk_level": "high" },
  "filters": {
    "screen_capture":        { "answer": "no",  "percentageOfConfidence": 95.0 },
    "printed_paper":         { "answer": "yes", "percentageOfConfidence": 98.0 },
    "superimposed_elements": { "answer": "no",  "percentageOfConfidence": 95.0 },
    "ai_altered":            { "answer": "no",  "percentageOfConfidence": 95.0 },
    "liveness":              { "answer": "yes", "percentageOfConfidence": 98.0 }
  }
}
```

**Fotografiado desde pantalla:**
```json
{
  "verdict": { "is_authentic": false, "overall_confidence": 2.0, "risk_level": "high" },
  "filters": {
    "screen_capture":        { "answer": "yes", "percentageOfConfidence": 98.0 },
    "printed_paper":         { "answer": "no",  "percentageOfConfidence": 98.0 },
    "superimposed_elements": { "answer": "no",  "percentageOfConfidence": 98.0 },
    "ai_altered":            { "answer": "no",  "percentageOfConfidence": 98.0 },
    "liveness":              { "answer": "yes", "percentageOfConfidence": 98.0 }
  }
}
```

---

## 6. Ejemplos de uso

### cURL

```bash
# Codificar imagen a Base64 y enviar a la API
IMAGE_B64=$(base64 -i cedula.jpg)

curl -s -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"${IMAGE_B64}\"}" | python -m json.tool
```

### Python

```python
import base64
import json
import requests

def verificar_cedula(ruta_imagen: str, api_url: str = "http://localhost:8000") -> dict:
    # Leer y codificar la imagen
    with open(ruta_imagen, "rb") as f:
        imagen_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Llamar a la API
    response = requests.post(
        f"{api_url}/verify",
        json={"image": imagen_b64},
        timeout=60,
    )
    response.raise_for_status()
    resultado = response.json()

    # Interpretar el resultado
    v = resultado["verdict"]
    print(f"¿Es auténtica?:    {'✅ SÍ' if v['is_authentic'] else '❌ NO'}")
    print(f"Confianza:         {v['overall_confidence']}%")
    print(f"Nivel de riesgo:   {v['risk_level'].upper()}")

    if not v["is_authentic"]:
        # Identificar qué filtro disparó
        for nombre, filtro in resultado["filters"].items():
            if filtro["answer"] == "yes":
                print(f"Fraude detectado:  {nombre} ({filtro['percentageOfConfidence']}%)")

    return resultado

# Uso
resultado = verificar_cedula("mi_cedula.jpg")
```

### JavaScript / Node.js

```javascript
const fs = require("fs");

async function verificarCedula(rutaImagen, apiUrl = "http://localhost:8000") {
  // Leer y codificar la imagen
  const imagenBuffer = fs.readFileSync(rutaImagen);
  const imagenB64 = imagenBuffer.toString("base64");

  // Llamar a la API
  const response = await fetch(`${apiUrl}/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imagenB64 }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`API error ${response.status}: ${error.detail}`);
  }

  const resultado = await response.json();
  const { verdict, filters } = resultado;

  console.log(`¿Es auténtica?: ${verdict.is_authentic ? "✅ SÍ" : "❌ NO"}`);
  console.log(`Confianza:      ${verdict.overall_confidence}%`);
  console.log(`Riesgo:         ${verdict.risk_level.toUpperCase()}`);

  if (!verdict.is_authentic) {
    for (const [nombre, filtro] of Object.entries(filters)) {
      if (filtro.answer === "yes") {
        console.log(`Fraude: ${nombre} (${filtro.percentageOfConfidence}%)`);
      }
    }
  }

  return resultado;
}

// Uso
verificarCedula("cedula.jpg").catch(console.error);
```

### PHP

```php
<?php
function verificarCedula(string $rutaImagen, string $apiUrl = "http://localhost:8000"): array {
    $imagenB64 = base64_encode(file_get_contents($rutaImagen));

    $ch = curl_init("$apiUrl/verify");
    curl_setopt_array($ch, [
        CURLOPT_POST => true,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT => 60,
        CURLOPT_HTTPHEADER => ["Content-Type: application/json"],
        CURLOPT_POSTFIELDS => json_encode(["image" => $imagenB64]),
    ]);

    $respuesta = curl_exec($ch);
    $httpCode  = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    curl_close($ch);

    if ($httpCode !== 200) {
        throw new RuntimeException("API error $httpCode: $respuesta");
    }

    $resultado = json_decode($respuesta, true);
    $veredicto = $resultado["verdict"];

    echo "¿Es auténtica?: " . ($veredicto["is_authentic"] ? "✅ SÍ" : "❌ NO") . "\n";
    echo "Confianza:      " . $veredicto["overall_confidence"] . "%\n";
    echo "Riesgo:         " . strtoupper($veredicto["risk_level"]) . "\n";

    return $resultado;
}

// Uso
$resultado = verificarCedula("cedula.jpg");
?>
```

---

## 7. Script CLI para pruebas

Incluido en el repositorio en `scripts/verify.py`. Permite probar el motor directamente sin levantar la API, útil para validar imágenes nuevas antes de integrar.

### Instalación del CLI

```bash
cd id-recognition
pip install -r requirements.txt
```

### Uso

```bash
# Verificar una imagen individual
python scripts/verify.py ruta/cedula.jpg

# Verificar todas las imágenes de una carpeta
python scripts/verify.py ruta/carpeta/

# Carpeta recursiva (incluye subcarpetas)
python scripts/verify.py ruta/carpeta/ --recursive

# Cambiar número de hilos paralelos (más rápido con carpetas grandes)
python scripts/verify.py ruta/carpeta/ --workers 8

# Salida en JSON (para procesar con scripts)
python scripts/verify.py ruta/cedula.jpg --json

# Guardar reporte completo en archivo
python scripts/verify.py ruta/carpeta/ --output reporte.json
```

### Ejemplo de salida en terminal

```
Inicializando motor Gemini...
  1 imagen(es) a procesar  •  4 workers en paralelo

  cedula_autentica.jpg  (8.2s)
    ✓ AUTÉNTICA  confianza=95%  riesgo=LOW
    ────── Detalle de filtros ──────
      1. Pantalla / cursor           no    95.0%
      2. Impresión en papel          no    95.0%
      3. Stickers / superpuestos     no    95.0%
      4. Generado por IA             no    95.0%
      5. Autenticidad consolidada    no    95.0%

══════════════════════════════════════════════════════════
  RESUMEN
══════════════════════════════════════════════════════════
  Total imágenes:       1
  Auténticas:           1
  Fraudes detectados:   0
  Tiempo promedio:      8.2s/imagen
```

### Código de salida del CLI

| Código | Significado |
|---|---|
| `0` | Todas las imágenes son auténticas |
| `1` | Al menos un fraude detectado o error en alguna imagen |

Útil para integrar en pipelines de CI/CD o scripts de automatización:
```bash
python scripts/verify.py cedula.jpg && echo "Aprobada" || echo "Rechazada"
```

---

## 8. Códigos de error

| HTTP | `detail` | Causa | Solución |
|---|---|---|---|
| `400` | `Image exceeds 10MB limit` | Imagen demasiado grande | Comprimir o redimensionar la imagen |
| `400` | `Unsupported format 'BMP'` | Formato no soportado | Convertir a JPEG, PNG o WEBP |
| `400` | `Image dimensions ... exceed 4096x4096` | Imagen demasiado grande en píxeles | Redimensionar a máximo 4096px |
| `400` | `Invalid base64` | El campo `image` no es Base64 válido | Verificar la codificación |
| `500` | `Internal error: ...` | Error interno del servidor | Revisar logs del servidor |

---

## 9. Despliegue con Docker

### Construcción de la imagen

```bash
docker build -t adamo-id:latest .
```

### Ejecución

```bash
docker run -d \
  --name adamo-id \
  -p 8000:8000 \
  -e GEMINI_API_KEY=AIza...tu_clave... \
  adamo-id:latest
```

### Con docker-compose

```yaml
version: "3.9"
services:
  adamo-id:
    image: adamo-id:latest
    ports:
      - "8000:8000"
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Levantar
GEMINI_API_KEY=AIza... docker-compose up -d

# Ver logs
docker-compose logs -f adamo-id
```

### Variables de entorno en producción

```env
GEMINI_API_KEY=AIza...           # Obligatorio
LOG_LEVEL=WARNING                # Reducir verbosidad en producción
```

---

## 10. Preguntas frecuentes

**¿Cuánto tarda en responder la API?**  
Entre 7 y 15 segundos por imagen. El tiempo varía según la resolución de la imagen y la latencia de la red hacia los servidores de Google Gemini.

**¿Qué pasa si la API de Gemini no está disponible?**  
La API retorna un error `500`. Se recomienda implementar reintentos con backoff exponencial en el cliente.

**¿La API almacena las imágenes enviadas?**  
No. Las imágenes se procesan en memoria y se descartan inmediatamente. No se guarda ninguna imagen ni dato personal.

**¿Funciona con documentos de otros países?**  
El motor está optimizado para cédulas colombianas (cédula de ciudadanía digital en policarbonato). Puede analizar pasaportes colombianos y otros documentos, pero con menor precisión en los detalles específicos del diseño.

**¿Por qué la confianza puede ser baja en imágenes auténticas?**  
Condiciones como poca luz, mucho desenfoque, ángulo inclinado o baja resolución pueden reducir la confianza. Siempre que `is_authentic: true` con `risk_level: "low"`, el documento se considera válido.

**¿Se puede ajustar el umbral de decisión?**  
Sí. El campo `options.confidence_threshold` en el request (entre 0.0 y 1.0) controla cuánta confianza mínima se requiere para marcar un documento como auténtico. Por defecto es `0.5`.

**¿Qué significa que `liveness` y otro filtro digan `"yes"` a la vez?**  
`liveness` siempre refleja el estado consolidado. Si cualquier filtro 1–4 disparó fraude, `liveness` también lo hará con la misma confianza. Nunca habrá una contradicción entre `liveness` y los otros filtros.

**¿Hay límite de llamadas por minuto?**  
El límite lo impone Google Gemini según el plan de API Key contratado. La API de ADAMO ID no añade límites adicionales. Se recomienda no superar 10 llamadas simultáneas.

---

## Contacto y soporte

Para reportar bugs o solicitar nuevas funcionalidades, abrir un issue en el repositorio de GitHub:  
**https://github.com/DTv2-1/id-recognition**
