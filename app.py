# pip install opencv-python mediapipe

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import tasks

# =============================================================================
#  Baixar modelo PoseLandmarker (Lite — otimizado para CPU)
# =============================================================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
# Salvar em caminho SEM caracteres acentuados — a lib C do MediaPipe
# não suporta caminhos Unicode (ex: "Detecção" com ç causa erro).
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
os.makedirs(_CACHE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(_CACHE_DIR, "pose_landmarker_lite.task")

if not os.path.exists(MODEL_PATH):
    print(f"Baixando modelo de {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Download concluído! Salvo em: {MODEL_PATH}")

# =============================================================================
#  Índices dos landmarks do MediaPipe Pose (33 pontos)
# =============================================================================
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# =============================================================================
#  Constantes de estado
# =============================================================================
NEUTRAL = "NEUTRAL"
A_UP_B_DOWN = "L_UP_R_DOWN"   # Pulso esquerdo acima, direito abaixo
B_UP_A_DOWN = "R_UP_L_DOWN"   # Pulso direito acima, esquerdo abaixo
BOTH_UP = "BOTH_UP"

STATE_COLORS = {
    NEUTRAL:     (160, 160, 160),
    A_UP_B_DOWN: (255, 140, 0),
    B_UP_A_DOWN: (0, 140, 255),
    BOTH_UP:     (0, 0, 255),
}

# =============================================================================
#  Parâmetros do detector de gesto
# =============================================================================
DEBOUNCE_SEC = 0.15
MAX_CYCLE_WINDOW_SEC = 4.0
RHYTHM_TOLERANCE = 0.50
MIN_TRANSITIONS = 4
FLASH_DURATION_SEC = 0.6
MIN_VISIBILITY = 0.5

# =============================================================================
#  Estado global (atualizado pelo callback do LIVE_STREAM)
# =============================================================================
latest_landmarks = None


def on_result(result, output_image, timestamp_ms):
    """Callback chamado pelo PoseLandmarker em modo LIVE_STREAM."""
    global latest_landmarks
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        latest_landmarks = result.pose_landmarks[0]
    else:
        latest_landmarks = None


# =============================================================================
#  Criar PoseLandmarker
# =============================================================================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=on_result,
)
landmarker = vision.PoseLandmarker.create_from_options(options)


# =============================================================================
#  Funções auxiliares
# =============================================================================
def classify_frame(landmarks):
    """Classifica o frame baseado na posição relativa dos pulsos."""
    wl = landmarks[LEFT_WRIST]
    wr = landmarks[RIGHT_WRIST]
    sl = landmarks[LEFT_SHOULDER]
    sr = landmarks[RIGHT_SHOULDER]
    hl = landmarks[LEFT_HIP]
    hr = landmarks[RIGHT_HIP]

    # Visibilidade mínima
    if wl.visibility < MIN_VISIBILITY or wr.visibility < MIN_VISIBILITY:
        return NEUTRAL

    shoulder_y = (sl.y + sr.y) / 2
    hip_y = (hl.y + hr.y) / 2
    chest_y = (shoulder_y + hip_y) / 2

    wl_y, wr_y = wl.y, wr.y

    if wl_y < shoulder_y and wr_y < shoulder_y:
        return BOTH_UP
    if wl_y < shoulder_y and wr_y > chest_y:
        return A_UP_B_DOWN
    if wr_y < shoulder_y and wl_y > chest_y:
        return B_UP_A_DOWN

    return NEUTRAL


def check_six_seven(trans):
    """Verifica padrão rítmico alternado (2 ciclos = 4 transições)."""
    if len(trans) < MIN_TRANSITIONS:
        return False

    last = list(trans)[-MIN_TRANSITIONS:]

    if not all(last[i][0] != last[i + 1][0] for i in range(MIN_TRANSITIONS - 1)):
        return False

    if (last[-1][1] - last[0][1]) > MAX_CYCLE_WINDOW_SEC:
        return False

    intervals = [last[i + 1][1] - last[i][1] for i in range(MIN_TRANSITIONS - 1)]
    avg = sum(intervals) / len(intervals)
    if avg == 0:
        return False

    return all(
        (1 - RHYTHM_TOLERANCE) * avg <= dt <= (1 + RHYTHM_TOLERANCE) * avg
        for dt in intervals
    )


# Conexões do esqueleto (subconjunto relevante para visualização do tronco/braços)
POSE_CONNECTIONS = [
    (11, 12),  # ombros
    (11, 13), (13, 15),  # braço esquerdo
    (12, 14), (14, 16),  # braço direito
    (11, 23), (12, 24),  # tronco
    (23, 24),  # quadril
]


def draw_skeleton(frame, landmarks):
    """Desenha o esqueleto sobre o frame."""
    h, w = frame.shape[:2]

    # Conexões
    for i, j in POSE_CONNECTIONS:
        p1 = landmarks[i]
        p2 = landmarks[j]
        if p1.visibility > 0.4 and p2.visibility > 0.4:
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), 2, cv2.LINE_AA)

    # Pontos
    for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
        lm = landmarks[idx]
        if lm.visibility > 0.4:
            x, y = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 255) if idx in (15, 16) else (50, 255, 50)
            cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)


def draw_reference_lines(frame, landmarks):
    """Desenha linhas de referência (ombros e peito)."""
    h, w = frame.shape[:2]
    sl = landmarks[LEFT_SHOULDER]
    sr = landmarks[RIGHT_SHOULDER]
    hl = landmarks[LEFT_HIP]
    hr = landmarks[RIGHT_HIP]

    shoulder_px = int((sl.y + sr.y) / 2 * h)
    chest_px = int(((sl.y + sr.y) / 2 + (hl.y + hr.y) / 2) / 2 * h)

    cv2.line(frame, (0, shoulder_px), (w, shoulder_px), (255, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Ombros", (w - 90, shoulder_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1, cv2.LINE_AA)

    cv2.line(frame, (0, chest_px), (w, chest_px), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Peito", (w - 75, chest_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)


def draw_hud(frame, state, gesture_count, buffer_len, flash_remaining):
    """Desenha o HUD sobre o frame."""
    h, w = frame.shape[:2]

    # Flash verde
    if flash_remaining > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
        alpha = 0.25 * (flash_remaining / FLASH_DURATION_SEC)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Fundo HUD
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (10, 10), (340, 140), (0, 0, 0), -1)
    cv2.addWeighted(hud_overlay, 0.5, frame, 0.5, 0, frame)

    # Contador
    cv2.putText(frame, f"Six Seven: {gesture_count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # Estado
    color = STATE_COLORS.get(state, (255, 255, 255))
    cv2.putText(frame, f"Estado: {state}",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Buffer
    cv2.putText(frame, f"Buffer: {buffer_len}/{MIN_TRANSITIONS}",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Instrução
    cv2.putText(frame, "'q' para sair",
                (w - 160, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


# =============================================================================
#  Loop principal
# =============================================================================
def main():
    global latest_landmarks

    webcam = cv2.VideoCapture(0)

    transitions = deque(maxlen=8)
    last_state = NEUTRAL
    last_transition_time = 0.0
    gesture_count = 0
    gesture_flash_time = 0.0
    frame_ts = 0

    print("Iniciando detector de gesto 'Six Seven'...")
    print("Pressione 'q' para sair.\n")

    while True:
        success, frame = webcam.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        now = time.time()
        state = NEUTRAL

        # Enviar frame para o PoseLandmarker (assíncrono)
        frame_ts += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, frame_ts)

        # Processar resultado mais recente
        if latest_landmarks is not None:
            lm = latest_landmarks

            draw_skeleton(frame, lm)
            draw_reference_lines(frame, lm)

            state = classify_frame(lm)

            # ── Máquina de estados ──
            if state in (A_UP_B_DOWN, B_UP_A_DOWN):
                if state != last_state and (now - last_transition_time) > DEBOUNCE_SEC:
                    transitions.append((state, now))
                    last_state = state
                    last_transition_time = now

                    if check_six_seven(transitions):
                        gesture_count += 1
                        gesture_flash_time = now
                        transitions.clear()
                        last_state = NEUTRAL
                        print(f"[SIX SEVEN] Gesto detectado! Total: {gesture_count}")

            elif state == BOTH_UP:
                transitions.clear()
                last_state = NEUTRAL

        # ── HUD ──
        flash_remaining = max(0, FLASH_DURATION_SEC - (now - gesture_flash_time))
        draw_hud(frame, state, gesture_count, len(transitions), flash_remaining)

        cv2.imshow("Six Seven Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\nTotal de gestos Six Seven detectados: {gesture_count}")


if __name__ == "__main__":
    main()
