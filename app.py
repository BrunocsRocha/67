# pip install opencv-python mediapipe numpy

import os
import json
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque
from datetime import date

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =============================================================================
#  Baixar modelo PoseLandmarker (Lite — otimizado para CPU)
# =============================================================================
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
os.makedirs(_CACHE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(_CACHE_DIR, "pose_landmarker_lite.task")
RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "six_seven_records.json")

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
#  Constantes de estado do gesto
# =============================================================================
NEUTRAL = "NEUTRAL"
A_UP_B_DOWN = "L_UP_R_DOWN"
B_UP_A_DOWN = "R_UP_L_DOWN"
BOTH_UP = "BOTH_UP"

STATE_COLORS = {
    NEUTRAL:     (160, 160, 160),
    A_UP_B_DOWN: (255, 140, 0),
    B_UP_A_DOWN: (0, 140, 255),
    BOTH_UP:     (0, 0, 255),
}

# =============================================================================
#  Constantes de tela
# =============================================================================
SCREEN_START = "START"
SCREEN_GAME = "GAME"
SCREEN_RESULTS = "RESULTS"

GAME_DURATION_SEC = 60

# =============================================================================
#  Parâmetros do detector de gesto (otimizados para sensibilidade)
# =============================================================================
DEBOUNCE_SEC = 0.08
MAX_CYCLE_WINDOW_SEC = 3.0
RHYTHM_TOLERANCE = 0.70
MIN_TRANSITIONS = 2
FLASH_DURATION_SEC = 0.6
MIN_VISIBILITY = 0.35
MIN_WRIST_MOVEMENT = 0.03  # Movimento mínimo de cada pulso (3% da altura normalizada)

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
#  Funções de detecção de gesto
# =============================================================================
def classify_frame(landmarks):
    """Classifica o frame baseado na posição relativa entre os dois pulsos.

    Quando o pulso esquerdo está mais alto (menor Y) que o direito → L_UP_R_DOWN.
    Quando o pulso direito está mais alto que o esquerdo → R_UP_L_DOWN.
    Margem de histerese para evitar oscilação quando estão na mesma altura.
    """
    wl = landmarks[LEFT_WRIST]
    wr = landmarks[RIGHT_WRIST]

    if wl.visibility < MIN_VISIBILITY or wr.visibility < MIN_VISIBILITY:
        return NEUTRAL

    # Margem de histerese (5% da altura normalizada)
    MARGIN = 0.05
    diff = wl.y - wr.y  # positivo = esquerdo mais baixo, negativo = esquerdo mais alto

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


# Conexões do esqueleto
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (11, 23),
    (12, 24), (23, 24),
]


# =============================================================================
#  Funções de desenho
# =============================================================================
def draw_skeleton(frame, landmarks):
    """Desenha o esqueleto sobre o frame."""
    h, w = frame.shape[:2]
    for i, j in POSE_CONNECTIONS:
        p1, p2 = landmarks[i], landmarks[j]
        if p1.visibility > 0.4 and p2.visibility > 0.4:
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (220, 220, 220), 2, cv2.LINE_AA)
    for idx in [11, 12, 13, 14, 15, 16, 23, 24]:
        lm = landmarks[idx]
        if lm.visibility > 0.4:
            x, y = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 255) if idx in (15, 16) else (50, 255, 50)
            cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)


def draw_reference_lines(frame, landmarks):
    """Desenha linhas de referência (ombros e peito)."""
    h, w = frame.shape[:2]
    sl, sr = landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
    hl, hr = landmarks[LEFT_HIP], landmarks[RIGHT_HIP]
    shoulder_px = int((sl.y + sr.y) / 2 * h)
    chest_px = int(((sl.y + sr.y) / 2 + (hl.y + hr.y) / 2) / 2 * h)
    cv2.line(frame, (0, shoulder_px), (w, shoulder_px), (255, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "Ombros", (w - 90, shoulder_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (0, chest_px), (w, chest_px), (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Peito", (w - 75, chest_px - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)


def draw_game_hud(frame, state, gesture_count, buffer_len, flash_remaining,
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
    color = STATE_COLORS.get(state, (255, 255, 255))
    cv2.putText(frame, f"Estado: {state}",
                (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Buffer
    cv2.putText(frame, f"Buffer: {buffer_len}/{MIN_TRANSITIONS}",
                (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

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

    # Título
    put_centered_text(frame, "RESULTADO", 55, 1.5, (0, 255, 255), 3)

    # Score do jogador
    put_centered_text(frame, f"{player_name}: {player_score} pontos",
                      100, 0.9, (255, 255, 255), 2)

    # Verificar se é novo recorde
    is_new_record = any(
        r["name"] == player_name and r["score"] == player_score for r in records[:10]
    )
    if is_new_record and player_score > 0:
        put_centered_text(frame, "NOVO RECORDE!", 135, 0.8, (0, 255, 0), 2)

    # Tabela de ranking
    table_x = (w - 400) // 2
    table_y = 160
    row_h = 32

    # Cabeçalho
    cv2.rectangle(frame, (table_x, table_y), (table_x + 400, table_y + row_h),
                  (50, 50, 80), -1)
    cv2.putText(frame, "#", (table_x + 10, table_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Nome", (table_x + 50, table_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Score", (table_x + 250, table_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Data", (table_x + 320, table_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    for i, rec in enumerate(records[:10]):
        ry = table_y + row_h * (i + 1)
        is_current = (rec["name"] == player_name and rec["score"] == player_score)

        # Fundo da linha
        bg_color = (40, 80, 40) if is_current else ((35, 35, 50) if i % 2 == 0 else (25, 25, 40))
        cv2.rectangle(frame, (table_x, ry), (table_x + 400, ry + row_h), bg_color, -1)

        text_color = (0, 255, 200) if is_current else (220, 220, 220)
        medal = ""
        if i == 0:
            medal = " [1st]"
        elif i == 1:
            medal = " [2nd]"
        elif i == 2:
            medal = " [3rd]"

        cv2.putText(frame, f"{i + 1}{medal}", (table_x + 10, ry + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)
        # Truncar nome se muito longo
        display_n = rec["name"][:12]
        cv2.putText(frame, display_n, (table_x + 50, ry + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, str(rec["score"]), (table_x + 260, ry + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        rec_date = rec.get("date", "")
        cv2.putText(frame, rec_date, (table_x + 320, ry + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA)

    # Botões
    btn_y = h - 80
    # Jogar Novamente
    btn1_x = (w // 2) - 200
    cv2.rectangle(frame, (btn1_x, btn_y), (btn1_x + 180, btn_y + 45),
                  (0, 160, 140), -1, cv2.LINE_AA)
    cv2.putText(frame, "ENTER: Jogar", (btn1_x + 15, btn_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    # Sair
    btn2_x = (w // 2) + 20
    cv2.rectangle(frame, (btn2_x, btn_y), (btn2_x + 150, btn_y + 45),
                  (0, 0, 160), -1, cv2.LINE_AA)
    cv2.putText(frame, "Q: Sair", (btn2_x + 25, btn_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)


# =============================================================================
#  Loop principal com máquina de estados de telas
# =============================================================================
def main():
    global latest_landmarks

    webcam = cv2.VideoCapture(0)

    # Estado das telas
    current_screen = SCREEN_START
    player_name = ""
    cursor_blink_time = time.time()

    # Estado do jogo
    transitions = deque(maxlen=8)
    last_state = NEUTRAL
    last_transition_time = 0.0
    gesture_count = 0
    gesture_flash_time = 0.0
    game_start_time = 0.0
    frame_ts = 0
    prev_wl_y = None  # Y do pulso esquerdo na última transição
    prev_wr_y = None  # Y do pulso direito na última transição

    # Recordes
    records = load_records()
    player_score = 0

    print("Iniciando Six Seven Challenge...")
    print("Pressione 'q' para sair.\n")

    while True:
        success, frame = webcam.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        now = time.time()

        # Enviar frame para o PoseLandmarker (sempre, para manter o feed ativo)
        frame_ts += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_image, frame_ts)

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
                # Processar detecção de pose
                if latest_landmarks is not None:
                    lm = latest_landmarks
                    draw_skeleton(frame, lm)
                    draw_reference_lines(frame, lm)
                    state = classify_frame(lm)

                    # Máquina de estados do gesto
                    cur_wl_y = lm[LEFT_WRIST].y
                    cur_wr_y = lm[RIGHT_WRIST].y

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
                                    print(f"[SIX SEVEN] +1! Total: {gesture_count}")


                flash_remaining = max(0, FLASH_DURATION_SEC - (now - gesture_flash_time))
                draw_game_hud(frame, state, gesture_count, len(transitions),
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

        cv2.imshow("Six Seven Challenge", frame)

    webcam.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\nSessao encerrada.")


if __name__ == "__main__":
    main()
