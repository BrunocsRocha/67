# Detector de Gesto "Six Seven" com MediaPipe Pose

## Descrição

Aplicação Python para detecção em tempo real do gesto **"Six Seven"** usando webcam. O gesto consiste em alternar os braços ritmicamente: um pulso sobe acima dos ombros enquanto o outro desce abaixo do peito, repetindo pelo menos 2 ciclos completos (~1.5s por ciclo).

O sistema utiliza **MediaPipe PoseLandmarker** (Tasks Vision API) para rastrear keypoints corporais e uma **máquina de estados** para identificar o padrão de alternância rítmica.

## Funcionalidades

- **Detecção de pose em tempo real** via MediaPipe Pose (modelo Lite, otimizado para CPU)
- **Máquina de estados** com 4 estados: `NEUTRAL`, `L_UP_R_DOWN`, `R_UP_L_DOWN`, `BOTH_UP`
- **Detecção de padrão rítmico**: 4 transições alternadas dentro de ~4s com intervalos regulares (±50%)
- **Filtro anti-falso-positivo**: movimentos com ambas as mãos levantadas simultaneamente são ignorados
- **Overlay visual**:
  - Esqueleto do tronco e braços com pontos nos pulsos (ciano)
  - Linhas de referência: ombros (amarelo) e peito (laranja)
  - HUD semi-transparente: contador, estado atual, buffer de transições
  - Flash verde ao detectar gesto
- **Espelhamento** para interação natural com a webcam

## Pré-requisitos

- Python 3.8+
- Webcam funcional

## Instalação

1. Clone o repositório ou baixe os arquivos
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso

```bash
python app.py
```

1. A webcam será ativada e você verá o feed com overlay de detecção de pose
2. Realize o gesto **Six Seven**: alterne os braços ritmicamente (um acima do ombro, outro abaixo do peito) por pelo menos 2 ciclos
3. O contador no canto superior esquerdo incrementará a cada detecção, com um flash verde na tela
4. Pressione **'q'** para sair

## Tecnologias

| Componente | Tecnologia |
|---|---|
| **Modelo de Pose** | MediaPipe PoseLandmarker (Lite) |
| **Visão Computacional** | OpenCV |
| **API** | MediaPipe Tasks Vision (`LIVE_STREAM`) |
| **Keypoints utilizados** | Pulsos (15, 16), Ombros (11, 12), Quadris (23, 24) |

## Algoritmo

```
NEUTRAL → L_UP_R_DOWN (pulso E acima ombros, D abaixo peito)
NEUTRAL → R_UP_L_DOWN (pulso D acima ombros, E abaixo peito)
L_UP_R_DOWN ↔ R_UP_L_DOWN (alternância < 1.5s)
Qualquer → NEUTRAL (BOTH_UP detectado → reset)

4 transições alternadas rítmicas (±50%) dentro de 4s → GESTO DETECTADO
```

## Estrutura do Projeto

```
├── app.py              # Aplicação principal
├── requirements.txt    # Dependências Python
├── README.md           # Este arquivo
```

## Créditos

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — Google AI Edge
- [OpenCV](https://opencv.org/) — Visão computacional
