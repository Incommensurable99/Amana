from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import insightface
from insightface.app import FaceAnalysis
import os
import base64

app = FastAPI()

# ================================
# CORS (para conexión con frontend)
# ================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# CONFIGURACIÓN MODELOS
# =====================
print(">>> Inicializando modelo de detección de rostros...")
app_insight = FaceAnalysis(allowed_modules=['detection'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))
print(">>> Modelo de rostros cargado correctamente.")

print(">>> Cargando modelo ResEmoteNet...")

# ---------------------
# MODELO ResEmoteNet
# ---------------------
class ResEmoteNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        from torchvision.models import resnet18
        self.base = resnet18(weights="IMAGENET1K_V1")
        self.base.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.base(x)

model = ResEmoteNet(num_classes=7)

# Si tienes un modelo entrenado:
# model.load_state_dict(torch.load("ResEmoteNet_pretrained.pth", map_location='cpu'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print(f">>> Modelo ResEmoteNet cargado correctamente en {device}.")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================
# ENDPOINT PRINCIPAL
# =====================
@app.post("/analizar")
async def analizar_video(file: UploadFile = File(...)):
    print("\n============================")
    print(f">>> Archivo recibido: {file.filename}")
    print("============================")

    # Guardar temporalmente el archivo
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f">>> Archivo guardado temporalmente en {temp_path}")

    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limite = total_frames // 2  # analizar mitad del video

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = f"procesado_{file.filename}.mp4"
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    emotion_stats = {e: 0 for e in EMOTIONS}
    max_conf_overall = 0
    dominant_emotion = None
    dominant_frame = None

    print(">>> Iniciando análisis de video...\n")

    # Analizar cada 2 frames para más velocidad
    frame_skip = 2  

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= limite:
            print("\n>>> Fin del video o alcanzado límite de frames.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # saltar frames para optimizar

        print(f">>> Procesando frame {frame_count}/{limite}")
        faces = app_insight.get(frame)
        print(f"    - Rostros detectados: {len(faces)}")

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"    ⚠️ Cara {i+1}: región vacía, se omite.")
                continue

            tensor = transform(crop).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)
                emotion_id = torch.argmax(probs).item()
                conf = probs[0][emotion_id].item()

            emotion = EMOTIONS[emotion_id]
            emotion_stats[emotion] += 1

            # Dibujar rectángulo y texto verde
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({conf*100:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            # Guardar frame con la emoción más dominante (mayor confianza)
            if conf > max_conf_overall:
                max_conf_overall = conf
                dominant_emotion = emotion
                dominant_frame = frame.copy()

        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_path)

    print("\n>>> Análisis de video completado.")
    print(f">>> Frames procesados: {frame_count}")

    total_faces = sum(emotion_stats.values())
    if total_faces == 0:
        print(">>> ❌ No se detectaron rostros en el video.")
        return {"mensaje": "No se detectaron rostros."}

    top_emotion = max(emotion_stats, key=emotion_stats.get)
    porcentaje = (emotion_stats[top_emotion] / total_faces) * 100

    distribucion_porcentual = {
        e: round((count / total_faces) * 100, 2) for e, count in emotion_stats.items()
    }

    print(f"\n>>> Emoción predominante: {top_emotion.upper()} ({porcentaje:.2f}%)")
    print(">>> Distribución de emociones:", distribucion_porcentual)

    # =============================
    # GUARDAR IMAGEN DOMINANTE
    # =============================
    image_base64 = None
    if dominant_frame is not None:
        cv2.putText(dominant_frame,
                    f"{dominant_emotion.upper()} ({max_conf_overall*100:.1f}%)",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        output_image_path = f"resultado_{dominant_emotion}.jpg"
        cv2.imwrite(output_image_path, dominant_frame)
        print(f"\n>>> Imagen representativa guardada: {output_image_path}")

        with open(output_image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        print(">>> Imagen convertida a base64 y lista para enviar.")

    print("\n>>> Preparando respuesta final...\n")

    return {
        "mensaje": "Análisis completado",
        "frames_procesados": frame_count,
        "rostros_detectados": total_faces,
        "emocion_predominante": top_emotion,
        "porcentaje": f"{porcentaje:.2f}%",
        "distribucion": distribucion_porcentual,
        "imagen_emocion": image_base64,
        "video_procesado": output_video_path
    }
