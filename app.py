# pip install opencv-python numpy ultralytics torch

import os
import sys
import json
import math
import cv2
import numpy as np
import time
from collections import deque
from datetime import date
from typing import Dict, Optional, Tuple

import torch
from ultralytics import YOLO

# =============================================================================
#  Configuracoes (antes em config.py)
# =============================================================================
WEBCAM_INDEX = 0
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
MIRROR_MODE = True

YOLO_MODEL = "yolov8n-pose.pt"
YOLO_IMGSZ = 640
YOLO_CONF = 0.40
YOLO_IOU = 0.50
YOLO_DEVICE = "auto"

MIN_KP_CONFIDENCE = 0.45
SMOOTHING_ALPHA = 0.45
HAND_BOX_SCALE = 0.30

COLOR_KEYPOINT = (0, 255, 200)
COLOR_SKELETON = (255, 180, 50)
COLOR_HAND_BOX = (80, 220, 255)
KEYPOINT_RADIUS = 6
SKELETON_THICKNESS = 3
HAND_BOX_THICKNESS = 2

# =============================================================================
#  PoseTracker (antes em tracker.py)
# =============================================================================
_COCO_KP_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle",
}
_REQUIRED_NAMES = {
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip",
}


class PoseTracker:
    def __init__(self):
        device = YOLO_DEVICE if YOLO_DEVICE != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[tracker] Loading {YOLO_MODEL} on device={device}")
        self.model = YOLO(YOLO_MODEL)
        self.device = device

    def process(self, frame):
        results = self.model.predict(
            source=frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
            iou=YOLO_IOU, device=self.device, verbose=False, max_det=5,
        )
        if not results or results[0].keypoints is None:
            return None
        kps_data = results[0].keypoints
        boxes = results[0].boxes
        if kps_data.xy is None or len(kps_data.xy) == 0:
            return None
        best_idx = self._select_person(boxes)
        if best_idx is None:
            return None
        xy = kps_data.xy[best_idx].cpu().numpy()
        conf = kps_data.conf[best_idx].cpu().numpy()
        out = {}
        for idx, name in _COCO_KP_NAMES.items():
            if name in _REQUIRED_NAMES:
                out[name] = (float(xy[idx, 0]), float(xy[idx, 1]), float(conf[idx]))
        return out

    @staticmethod
    def _select_person(boxes):
        if boxes is None or len(boxes) == 0:
            return None
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        max_area = areas.max() if areas.max() > 0 else 1.0
        scores = confs * 0.6 + (areas / max_area) * 0.4
        return int(np.argmax(scores))


# =============================================================================
#  Utils (antes em utils.py)
# =============================================================================
def filter_keypoints(raw):
    return {name: (x, y) for name, (x, y, c) in raw.items() if c >= MIN_KP_CONFIDENCE}


class KeypointSmoother:
    def __init__(self, alpha=SMOOTHING_ALPHA):
        self.alpha = alpha
        self._state = {}

    def smooth(self, keypoints):
        result = {}
        for name, (x, y) in keypoints.items():
            cur = np.array([x, y], dtype=np.float64)
            if name in self._state:
                smoothed = self.alpha * self._state[name] + (1.0 - self.alpha) * cur
            else:
                smoothed = cur
            self._state[name] = smoothed
            result[name] = (float(smoothed[0]), float(smoothed[1]))
        self._state = {k: v for k, v in self._state.items() if k in keypoints}
        return result

    def reset(self):
        self._state.clear()


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit_vector(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    mag = math.hypot(dx, dy)
    if mag < 1e-6:
        return None
    return (dx / mag, dy / mag)


def shoulder_distance(kps):
    ls, rs = kps.get("left_shoulder"), kps.get("right_shoulder")
    if ls is None or rs is None:
        return None
    return _distance(ls, rs)


def estimate_hand_box(wrist, elbow, body_scale, scale=HAND_BOX_SCALE):
    half = max(int(body_scale * scale / 2), 8)
    uv = _unit_vector(elbow, wrist)
    if uv is not None:
        cx = wrist[0] + uv[0] * half * 0.5
        cy = wrist[1] + uv[1] * half * 0.5
    else:
        cx, cy = wrist
    return (int(cx - half), int(cy - half)), (int(cx + half), int(cy + half))

# =============================================================================
#  Caminhos
# =============================================================================
RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "six_seven_records.json")

# =============================================================================
#  Constantes de tela
# =============================================================================
SCREEN_START = "START"
SCREEN_GAME = "GAME"
SCREEN_RESULTS = "RESULTS"

GAME_DURATION_SEC = 60
FLASH_DURATION_SEC = 0.6

# =============================================================================
#  Parâmetros do detector de gesto (original wrist-based)
# =============================================================================
NEUTRAL = "NEUTRAL"
A_UP_B_DOWN = "L_UP_R_DOWN"
B_UP_A_DOWN = "R_UP_L_DOWN"

DEBOUNCE_SEC = 0.08
MAX_CYCLE_WINDOW_SEC = 3.0
RHYTHM_TOLERANCE = 0.70
MIN_TRANSITIONS = 2
MIN_WRIST_MOVEMENT = 3.0  # Movimento mínimo em pixels

STATE_COLORS = {
    NEUTRAL:     (160, 160, 160),
    A_UP_B_DOWN: (255, 140, 0),
    B_UP_A_DOWN: (0, 140, 255),
}

# =============================================================================
#  Conexões do esqueleto (nomes COCO usados pelo YOLO)
# =============================================================================
_SKELETON_PAIRS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


# =============================================================================
#  Funções de detecção de gesto (original — baseada em pulsos)
# =============================================================================
def classify_frame_yolo(kps, frame_h):
    """Classifica o frame baseado na posição Y relativa entre os dois pulsos.

    Usa keypoints YOLO {name: (x, y)} com coordenadas em pixels.
    Normaliza pelo frame_h para manter a mesma lógica do original.
    """
    wl = kps.get("left_wrist")
    wr = kps.get("right_wrist")
    if wl is None or wr is None:
        return NEUTRAL

    # Normalizar Y pelo height do frame (como o MediaPipe fazia)
    wl_y = wl[1] / frame_h
    wr_y = wr[1] / frame_h

    # Margem de histerese (5% da altura normalizada)
    MARGIN = 0.05
    diff = wl_y - wr_y  # positivo = esquerdo mais baixo

    if diff < -MARGIN:
        return A_UP_B_DOWN   # Pulso esquerdo mais alto
    if diff > MARGIN:
        return B_UP_A_DOWN   # Pulso direito mais alto

    return NEUTRAL


def check_six_seven(trans):
    """Verifica padrão rítmico alternado."""
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


# =============================================================================
#  Sistema de Recordes (Top 10)
# =============================================================================
def load_records():
    """Carrega recordes do JSON. Retorna lista ordenada por score desc."""
    if not os.path.exists(RECORDS_PATH):
        return []
    try:
        with open(RECORDS_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        records.sort(key=lambda r: r.get("score", 0), reverse=True)
        return records[:10]
    except (json.JSONDecodeError, IOError):
        return []


def save_records(records):
    """Salva recordes no JSON."""
    with open(RECORDS_PATH, "w", encoding="utf-8") as f:
        json.dump(records[:10], f, ensure_ascii=False, indent=2)


def update_records(records, name, score):
    """Insere novo recorde na posição correta, mantém top 10."""
    entry = {"name": name, "score": score, "date": str(date.today())}
    records.append(entry)
    records.sort(key=lambda r: r.get("score", 0), reverse=True)
    return records[:10]


# =============================================================================
#  Funções de desenho — esqueleto YOLO (coordenadas em pixels)
# =============================================================================
def _int_pt(p):
    return (int(round(p[0])), int(round(p[1])))


def draw_skeleton_yolo(frame, kps):
    """Desenha o esqueleto usando keypoints YOLO {name: (x,y)}."""
    for a_name, b_name in _SKELETON_PAIRS:
        a = kps.get(a_name)
        b = kps.get(b_name)
        if a is not None and b is not None:
            cv2.line(frame, _int_pt(a), _int_pt(b),
                     COLOR_SKELETON, SKELETON_THICKNESS, cv2.LINE_AA)

    for name, pt in kps.items():
        if "nose" in name or "eye" in name or "ear" in name:
            continue  # Nao desenhar ponto no rosto
        color = COLOR_HAND_BOX if "wrist" in name else COLOR_KEYPOINT
        cv2.circle(frame, _int_pt(pt), KEYPOINT_RADIUS, color, -1, cv2.LINE_AA)


def draw_hand_boxes_yolo(frame, kps, body_scale):
    """Desenha caixas estimadas das maos."""
    for side in ("left", "right"):
        wrist = kps.get(f"{side}_wrist")
        elbow = kps.get(f"{side}_elbow")
        if wrist is not None and elbow is not None:
            tl, br = estimate_hand_box(wrist, elbow, body_scale)
            cv2.rectangle(frame, tl, br, COLOR_HAND_BOX,
                          HAND_BOX_THICKNESS, cv2.LINE_AA)


# =============================================================================
#  Funções de desenho — HUD e telas (mantidas do app.py original)
# =============================================================================
def draw_game_hud(frame, state_label, gesture_count, flash_remaining,
                  remaining_secs, player_name):
    """Desenha o HUD durante o jogo."""
    h, w = frame.shape[:2]

    # Flash verde
    if flash_remaining > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 255, 0), -1)
        alpha = 0.25 * (flash_remaining / FLASH_DURATION_SEC)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Fundo HUD esquerdo
    hud_overlay = frame.copy()
    cv2.rectangle(hud_overlay, (10, 10), (340, 160), (0, 0, 0), -1)
    cv2.addWeighted(hud_overlay, 0.5, frame, 0.5, 0, frame)

    # Nome do jogador
    cv2.putText(frame, f"Jogador: {player_name}",
                (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)

    # Contador
    cv2.putText(frame, f"Six Seven: {gesture_count}",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # Estado
    color = STATE_COLORS.get(state_label, (255, 255, 255))
    cv2.putText(frame, f"Estado: {state_label}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Timer (canto superior direito)
    mins = int(remaining_secs) // 60
    secs = int(remaining_secs) % 60
    timer_text = f"{mins:01d}:{secs:02d}"
    timer_color = (0, 255, 255) if remaining_secs > 10 else (0, 0, 255)

    # Fundo timer
    timer_overlay = frame.copy()
    cv2.rectangle(timer_overlay, (w - 160, 10), (w - 10, 70), (0, 0, 0), -1)
    cv2.addWeighted(timer_overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, timer_text,
                (w - 145, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, timer_color, 3, cv2.LINE_AA)

    # Instrução
    cv2.putText(frame, "'q' para sair",
                (w - 160, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)


def put_centered_text(frame, text, y, scale, color, thickness):
    """Escreve texto centralizado horizontalmente."""
    w = frame.shape[1]
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    x = (w - text_size[0]) // 2
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_start_screen(frame, player_name, cursor_visible):
    """Desenha a tela de início sobre o feed da webcam."""
    h, w = frame.shape[:2]

    # Overlay escuro
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 15, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Título
    put_centered_text(frame, "SIX SEVEN", h // 2 - 120, 2.0, (0, 255, 255), 4)
    put_centered_text(frame, "CHALLENGE", h // 2 - 70, 1.5, (0, 200, 220), 3)

    # Campo de nome
    label_text = "Nome do Jogador:"
    put_centered_text(frame, label_text, h // 2 - 15, 0.7, (200, 200, 200), 1)

    # Caixa de texto
    box_w, box_h = 350, 50
    box_x = (w - box_w) // 2
    box_y = h // 2
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h),
                  (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (box_x + 2, box_y + 2), (box_x + box_w - 2, box_y + box_h - 2),
                  (30, 30, 30), -1)

    display_name = player_name + ("|" if cursor_visible else "")
    cv2.putText(frame, display_name, (box_x + 15, box_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # Botão START
    if len(player_name) > 0:
        btn_w, btn_h = 250, 55
        btn_x = (w - btn_w) // 2
        btn_y = h // 2 + 80
        cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h),
                      (0, 200, 180), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h),
                      (0, 255, 255), 2, cv2.LINE_AA)
        put_centered_text(frame, "PRESSIONE ENTER", btn_y + 37, 0.7, (0, 0, 0), 2)
    else:
        put_centered_text(frame, "Digite seu nome para iniciar", h // 2 + 110, 0.55, (120, 120, 120), 1)

    # Regras
    put_centered_text(frame, f"Tempo: {GAME_DURATION_SEC}s | Alterne os bracos ritmicamente!",
                      h - 40, 0.5, (140, 140, 140), 1)


def draw_results_screen(frame, records, player_name, player_score):
    """Desenha a tela de resultados / ranking."""
    h, w = frame.shape[:2]

    # Overlay escuro
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # Calcular layout baseado na altura da tela
    header_zone_h = int(h * 0.25)
    footer_zone_h = 100
    table_zone_h = h - header_zone_h - footer_zone_h

    # Título
    title_y = int(header_zone_h * 0.35)
    put_centered_text(frame, "RESULTADO", title_y, 1.5, (0, 255, 255), 3)

    # Score do jogador
    score_y = int(header_zone_h * 0.60)
    put_centered_text(frame, f"{player_name}: {player_score} pontos",
                      score_y, 0.9, (255, 255, 255), 2)

    # Verificar se é novo recorde
    is_new_record = any(
        r["name"] == player_name and r["score"] == player_score for r in records[:10]
    )
    if is_new_record and player_score > 0:
        record_y = int(header_zone_h * 0.82)
        put_centered_text(frame, "NOVO RECORDE!", record_y, 0.8, (0, 255, 0), 2)

    # Tabela de ranking
    num_records = min(len(records), 10)
    table_w = min(750, w - 40)
    row_h = min(36, max(24, table_zone_h // (num_records + 2)))
    table_x = (w - table_w) // 2
    table_y = header_zone_h + 10

    # Definir colunas com espaço suficiente para rank + medalhas
    col_rank = table_x + 10
    col_nome = table_x + int(table_w * 0.18)
    col_score = table_x + int(table_w * 0.58)
    col_data = table_x + int(table_w * 0.72)

    # Cabeçalho
    cv2.rectangle(frame, (table_x, table_y), (table_x + table_w, table_y + row_h),
                  (50, 50, 80), -1)
    cv2.putText(frame, "#", (col_rank, table_y + row_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Nome", (col_nome, table_y + row_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Score", (col_score, table_y + row_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Data", (col_data, table_y + row_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    for i, rec in enumerate(records[:10]):
        ry = table_y + row_h * (i + 1)
        if ry + row_h > h - footer_zone_h:
            break
        is_current = (rec["name"] == player_name and rec["score"] == player_score)

        # Fundo da linha
        bg_color = (40, 80, 40) if is_current else ((35, 35, 50) if i % 2 == 0 else (25, 25, 40))
        cv2.rectangle(frame, (table_x, ry), (table_x + table_w, ry + row_h), bg_color, -1)

        text_color = (0, 255, 200) if is_current else (220, 220, 220)

        # Rank e medalha em colunas separadas
        rank_text = str(i + 1)
        medal_text = ""
        if i == 0:
            medal_text = "[1st]"
        elif i == 1:
            medal_text = "[2nd]"
        elif i == 2:
            medal_text = "[3rd]"

        cv2.putText(frame, rank_text, (col_rank, ry + row_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        if medal_text:
            medal_color = (0, 215, 255) if i == 0 else ((192, 192, 192) if i == 1 else (80, 127, 205))
            cv2.putText(frame, medal_text, (col_rank + 30, ry + row_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, medal_color, 1, cv2.LINE_AA)

        # Nome — permitir até 15 caracteres
        display_n = rec["name"][:15]
        cv2.putText(frame, display_n, (col_nome, ry + row_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, str(rec["score"]), (col_score, ry + row_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        rec_date = rec.get("date", "")
        cv2.putText(frame, rec_date, (col_data, ry + row_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    # Botões
    btn_y = h - 80
    btn1_x = (w // 2) - 200
    cv2.rectangle(frame, (btn1_x, btn_y), (btn1_x + 180, btn_y + 45),
                  (0, 160, 140), -1, cv2.LINE_AA)
    cv2.putText(frame, "ENTER: Jogar", (btn1_x + 15, btn_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    btn2_x = (w // 2) + 20
    cv2.rectangle(frame, (btn2_x, btn_y), (btn2_x + 150, btn_y + 45),
                  (0, 0, 160), -1, cv2.LINE_AA)
    cv2.putText(frame, "Q: Sair", (btn2_x + 25, btn_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)


# =============================================================================
#  Loop principal com máquina de estados de telas + YOLO Pose tracking
# =============================================================================
def main():
    # ── Inicializar YOLO Pose tracker + smoother ──
    tracker = PoseTracker()
    smoother = KeypointSmoother(alpha=SMOOTHING_ALPHA)

    webcam = cv2.VideoCapture(WEBCAM_INDEX)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    # Configurar janela em tela cheia
    window_name = "Six Seven Challenge"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Estado das telas
    current_screen = SCREEN_START
    player_name = ""
    cursor_blink_time = time.time()

    # Estado do jogo (original wrist-based)
    transitions = deque(maxlen=8)
    last_state = NEUTRAL
    last_transition_time = 0.0
    gesture_count = 0
    gesture_flash_time = 0.0
    game_start_time = 0.0
    prev_wl_y = None  # Y do pulso esquerdo na última transição
    prev_wr_y = None  # Y do pulso direito na última transição

    # Recordes
    records = load_records()
    player_score = 0

    print("Iniciando Six Seven Challenge (YOLO Pose)...")
    print("Pressione 'q' para sair.\n")

    while True:
        success, frame = webcam.read()
        if not success:
            continue

        if MIRROR_MODE:
            frame = cv2.flip(frame, 1)

        # Redimensionar frame para preencher a tela
        rect = cv2.getWindowImageRect(window_name)
        screen_w = int(rect[2]) if rect[2] > 0 else frame.shape[1]
        screen_h = int(rect[3]) if rect[3] > 0 else frame.shape[0]
        if screen_w > 0 and screen_h > 0:
            frame = cv2.resize(frame, (screen_w, screen_h))

        now = time.time()
        h_frame = frame.shape[0]

        # ── YOLO Pose inference (sempre, para manter feed ativo) ──
        raw_kps = tracker.process(frame)
        kps = None
        if raw_kps is not None:
            kps = filter_keypoints(raw_kps)
            kps = smoother.smooth(kps)
        else:
            smoother.reset()

        key = cv2.waitKey(1) & 0xFF

        # =================================================================
        #  TELA DE INÍCIO
        # =================================================================
        if current_screen == SCREEN_START:
            cursor_visible = (int((now - cursor_blink_time) * 2) % 2 == 0)
            draw_start_screen(frame, player_name, cursor_visible)

            if key == 13 and len(player_name) > 0:  # Enter
                current_screen = SCREEN_GAME
                game_start_time = time.time()
                gesture_count = 0
                gesture_flash_time = 0.0
                transitions.clear()
                last_state = NEUTRAL
                last_transition_time = 0.0
                prev_wl_y = None
                prev_wr_y = None
                smoother.reset()
                print(f"Jogo iniciado! Jogador: {player_name}")
            elif key == 8:  # Backspace
                player_name = player_name[:-1]
            elif key == ord("q") and len(player_name) == 0:
                break
            elif 32 <= key <= 126 and len(player_name) < 15:
                player_name += chr(key)

        # =================================================================
        #  TELA DE JOGO
        # =================================================================
        elif current_screen == SCREEN_GAME:
            remaining = GAME_DURATION_SEC - (now - game_start_time)
            state = NEUTRAL

            if remaining <= 0:
                # Tempo esgotado — transicionar para resultados
                player_score = gesture_count
                records = update_records(records, player_name, player_score)
                save_records(records)
                current_screen = SCREEN_RESULTS
                print(f"Tempo esgotado! {player_name}: {player_score} pontos")
            else:
                # Processar detecção de pose com YOLO
                if kps is not None and len(kps) > 0:
                    # Desenhar esqueleto e caixas das mãos
                    draw_skeleton_yolo(frame, kps)
                    body_scale = shoulder_distance(kps) or 150.0
                    draw_hand_boxes_yolo(frame, kps, body_scale)

                    # Classificar frame (lógica original baseada em pulsos)
                    state = classify_frame_yolo(kps, h_frame)

                    # Máquina de estados do gesto (original)
                    wl = kps.get("left_wrist")
                    wr = kps.get("right_wrist")
                    if wl is not None and wr is not None:
                        cur_wl_y = wl[1]
                        cur_wr_y = wr[1]

                        if state in (A_UP_B_DOWN, B_UP_A_DOWN):
                            if state != last_state and (now - last_transition_time) > DEBOUNCE_SEC:
                                # Verificar se AMBOS os pulsos se movimentaram
                                both_moved = True
                                if prev_wl_y is not None and prev_wr_y is not None:
                                    wl_delta = abs(cur_wl_y - prev_wl_y)
                                    wr_delta = abs(cur_wr_y - prev_wr_y)
                                    both_moved = (wl_delta >= MIN_WRIST_MOVEMENT and
                                                  wr_delta >= MIN_WRIST_MOVEMENT)

                                if both_moved:
                                    transitions.append((state, now))
                                    last_state = state
                                    last_transition_time = now
                                    prev_wl_y = cur_wl_y
                                    prev_wr_y = cur_wr_y

                                    if check_six_seven(transitions):
                                        gesture_count += 1
                                        gesture_flash_time = now
                                        transitions.clear()
                                        last_state = NEUTRAL
                                        prev_wl_y = None
                                        prev_wr_y = None

                flash_remaining = max(0, FLASH_DURATION_SEC - (now - gesture_flash_time))
                draw_game_hud(frame, state, gesture_count,
                              flash_remaining, max(0, remaining), player_name)

            if key == ord("q"):
                break

        # =================================================================
        #  TELA DE RESULTADOS
        # =================================================================
        elif current_screen == SCREEN_RESULTS:
            draw_results_screen(frame, records, player_name, player_score)

            if key == 13:  # Enter — jogar novamente
                current_screen = SCREEN_START
                player_name = ""
                cursor_blink_time = time.time()
            elif key == ord("q"):
                break

        cv2.imshow(window_name, frame)

    webcam.release()
    cv2.destroyAllWindows()
    print(f"\nSessao encerrada.")


if __name__ == "__main__":
    main()
