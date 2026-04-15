"""
Filtro 5 — Detección de Vida Real (Liveness)
Sistema de scoring compuesto basado en 6 señales independientes:

1. Resultado Filtro 1 (no es pantalla)           — 25%
2. Resultado Filtro 2 (no es impresión)          — 20%
3. Anti-spoofing del rostro (MiniFASNetV2)       — 20%
4. Análisis de perspectiva y bordes (OpenCV)      — 15%
5. Detección de manos/dedos (MediaPipe Hands)     — 10%
6. Análisis de metadatos EXIF (Pillow)           — 10%
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import Base as ExifBase

from app.config import FACE_ANTISPOOF_INPUT_SIZE, FACE_ANTISPOOF_REAL_LABEL, LIVENESS_WEIGHTS, weights_dir
from app.schemas import FilterResult

logger = logging.getLogger(__name__)


# ── Señal 3: Face Anti-Spoofing (MiniFASNetV2) ─────────────────
def _load_face_antispoof_onnx() -> object | None:
    """Carga modelo ONNX de MiniFASNetV2 si existe."""
    try:
        import onnxruntime as ort

        model_path = weights_dir() / "minifasnetv2_antispoof.onnx"
        if model_path.exists():
            session = ort.InferenceSession(
                str(model_path),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            logger.info("Face anti-spoof ONNX loaded from %s", model_path)
            return session
        logger.warning("No ONNX model at %s — face anti-spoof disabled.", model_path)
    except ImportError:
        logger.warning("onnxruntime not installed — face anti-spoof disabled.")
    return None


def predict_face_antispoof(
    face_crop: np.ndarray,
    session: object | None,
) -> float:
    """
    Ejecuta anti-spoofing sobre un recorte de rostro.
    Retorna score 0-1 donde 1 = cara real.
    """
    if session is None:
        return 0.5  # sin modelo → neutral

    import onnxruntime as ort
    from scipy.special import softmax

    h, w = FACE_ANTISPOOF_INPUT_SIZE
    resized = cv2.resize(face_crop, (w, h))
    # HWC → CHW, float32, rango [0, 255] (NO normalizar /255)
    tensor = resized.transpose((2, 0, 1)).astype(np.float32)
    tensor = np.expand_dims(tensor, axis=0)

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0]
    probs = softmax(logits, axis=1)[0]
    # label 1 = real face
    return float(probs[FACE_ANTISPOOF_REAL_LABEL])


# ── Señal 4: Perspectiva y bordes (OpenCV) ─────────────────────
def analyze_perspective(image_cv: np.ndarray) -> float:
    """
    Analiza si el documento tiene distorsión de perspectiva natural
    (indica que fue sostenido y fotografiado, no escaneado).
    Retorna score 0-1 donde 1 = perspectiva natural detectada.
    """
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Buscar líneas con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=50, maxLineGap=10)
    if lines is None or len(lines) < 4:
        return 0.3  # pocas líneas → no concluyente

    # Calcular ángulos de las líneas
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angles.append(angle % 180)

    angles = np.array(angles)

    # Un documento perfectamente recto tiene ángulos ~0° y ~90°
    # Un documento sostenido tiene ángulos ligeramente desviados
    near_0 = np.sum((angles < 10) | (angles > 170))
    near_90 = np.sum((angles > 80) & (angles < 100))
    total = len(angles)

    perfect_ratio = (near_0 + near_90) / total if total > 0 else 1.0

    # Si es "demasiado perfecto" → probablemente escaneado/captura de pantalla
    # Si tiene ligera desviación → probablemente sostenido
    if perfect_ratio > 0.85:
        return 0.3  # muy recto → sospechoso
    elif perfect_ratio > 0.5:
        return 0.8  # algo de perspectiva → bien
    else:
        return 0.5  # muy desordenado → ambiguo


# ── Señal 5: Detección de manos (MediaPipe) ────────────────────
def detect_hands(image_cv: np.ndarray) -> float:
    """
    Detecta si hay manos/dedos visibles en la imagen.
    Retorna score 0-1 donde 1 = manos detectadas (bueno para liveness).
    """
    try:
        import mediapipe as mp

        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.3,
        ) as hands:
            rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                # Verificar que al menos una mano está cerca de los bordes
                h, w, _ = image_cv.shape
                for hand in results.multi_hand_landmarks:
                    for lm in hand.landmark:
                        # Dedos cerca de los bordes del documento
                        if lm.x < 0.1 or lm.x > 0.9 or lm.y < 0.1 or lm.y > 0.9:
                            return 0.9  # mano en borde → muy bueno
                return 0.6  # mano detectada pero no en bordes
    except ImportError:
        logger.warning("MediaPipe not available — hand detection disabled.")
    except Exception as e:
        logger.debug("Hand detection error: %s", e)

    return 0.3  # sin detección


# ── Señal 6: Análisis EXIF ─────────────────────────────────────
def analyze_exif(image: Image.Image) -> float:
    """
    Analiza metadatos EXIF para indicios de autenticidad.
    Retorna score 0-1 donde 1 = metadatos consistentes con foto real.
    """
    score = 0.0
    checks = 0

    try:
        exif = image.getexif()
    except Exception:
        return 0.3  # sin EXIF → neutral negativo

    if not exif:
        return 0.2  # screenshots y ediciones suelen borrar EXIF

    # ¿Tiene modelo de cámara?
    if ExifBase.Make in exif or ExifBase.Model in exif:
        score += 1.0
        checks += 1
    else:
        checks += 1

    # ¿Tiene timestamp?
    if ExifBase.DateTime in exif or ExifBase.DateTimeOriginal in exif:
        score += 1.0
        checks += 1
    else:
        checks += 1

    # ¿Tiene dimensiones originales?
    if ExifBase.ExifImageWidth in exif and ExifBase.ExifImageHeight in exif:
        score += 0.8
        checks += 1
    else:
        checks += 1

    # ¿Software de edición mencionado?
    software = exif.get(ExifBase.Software, "")
    if isinstance(software, str) and any(kw in software.lower() for kw in ["photoshop", "gimp", "paint"]):
        score -= 0.5
        checks += 1
    else:
        score += 0.5
        checks += 1

    return max(0.0, min(1.0, score / checks if checks > 0 else 0.3))


# ── Detección de doble compresión JPEG ─────────────────────────
def check_double_compression(image: Image.Image) -> float:
    """
    Verifica indicios de doble compresión JPEG (señal de edición).
    Retorna score 0-1 donde 1 = sin doble compresión (bueno).
    """
    buf = io.BytesIO()
    try:
        image.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        recompressed = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
        original = np.asarray(image.convert("RGB").resize(recompressed.shape[1::-1]), dtype=np.float32)

        diff = np.abs(original - recompressed)
        mean_diff = float(np.mean(diff))

        # Doble compresión produce artefactos detectables
        if mean_diff > 8.0:
            return 0.3  # probable doble compresión
        elif mean_diff > 4.0:
            return 0.6
        else:
            return 0.9
    except Exception:
        return 0.5


# ── Detector de rostro simple (Haar cascade) ──────────────────
def extract_face_crop(image_cv: np.ndarray) -> np.ndarray | None:
    """Extrae el recorte del rostro más grande de la imagen."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

    # Tomar el rostro más grande
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # Expandir un poco el recorte
    pad = int(0.3 * max(w, h))
    ih, iw = image_cv.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(iw, x + w + pad)
    y2 = min(ih, y + h + pad)
    return image_cv[y1:y2, x1:x2]


# ── Detector Liveness Compuesto ────────────────────────────────
@dataclass
class LivenessDetector:
    """Filtro 5: ¿La foto fue tomada en tiempo real por un usuario real?"""

    device: str = "cpu"
    _antispoof_session: object | None = field(default=None, repr=False)

    def load_model(self, device: str = "cpu") -> None:
        self.device = device
        self._antispoof_session = _load_face_antispoof_onnx()

    def predict(
        self,
        image: Image.Image,
        screen_result: FilterResult,
        printed_result: FilterResult,
    ) -> FilterResult:
        """Calcula scoring compuesto de las 6 señales."""
        image_cv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Señal 1: No es pantalla (del Filtro 1)
        if screen_result.answer == "no":
            not_screen = screen_result.percentageOfConfidence / 100.0
        else:
            not_screen = 1.0 - (screen_result.percentageOfConfidence / 100.0)

        # Señal 2: No es impresión (del Filtro 2)
        if printed_result.answer == "no":
            not_printed = printed_result.percentageOfConfidence / 100.0
        else:
            not_printed = 1.0 - (printed_result.percentageOfConfidence / 100.0)

        # Señal 3: Face anti-spoofing
        face_crop = extract_face_crop(image_cv)
        face_live = predict_face_antispoof(face_crop, self._antispoof_session) if face_crop is not None else 0.5

        # Señal 4: Perspectiva
        perspective_valid = analyze_perspective(image_cv)

        # Señal 5: Manos
        hand_detected = detect_hands(image_cv)

        # Señal 6: EXIF
        exif_score = analyze_exif(image)
        compression_score = check_double_compression(image)
        exif_valid = 0.6 * exif_score + 0.4 * compression_score

        # Scoring ponderado
        w = LIVENESS_WEIGHTS
        score = (
            w.not_screen * not_screen
            + w.not_printed * not_printed
            + w.face_live * face_live
            + w.perspective_valid * perspective_valid
            + w.hand_detected * hand_detected
            + w.exif_valid * exif_valid
        )
        score = max(0.0, min(1.0, score))

        logger.debug(
            "Liveness signals: screen=%.2f print=%.2f face=%.2f persp=%.2f hand=%.2f exif=%.2f → %.2f",
            not_screen, not_printed, face_live, perspective_valid, hand_detected, exif_valid, score,
        )

        is_live = score > 0.5
        confidence = score if is_live else 1.0 - score
        return FilterResult(
            answer="yes" if is_live else "no",
            percentageOfConfidence=round(confidence * 100, 1),
        )
