from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import torch
import insightface
from insightface.app import FaceAnalysis
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os
import base64

app = FastAPI()

# ==========================================
# CORS (permite conexión con tu frontend React)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MODELOS: Detección facial + Emociones
# ==========================================
print(">>> Cargando detector de rostros (InsightFace)...")
app_insight = FaceAnalysis(allowed_modules=['detection'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Detector de rostros listo.")

print(">>> Cargando modelo de emociones (ViT-Face-Expression)...")
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"✅ Modelo ViT-Face-Expression cargado en {device}.")

# ==========================================
# ENDPOINT PRINCIPAL
# ==========================================
@app.post("/analizar")
async def analizar_video(file: UploadFile = File(...)):
    print("\n============================")
    print(f">>> Archivo recibido: {file.filename}")
    print("============================")

    # Guardar el archivo temporalmente
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f">>> Archivo guardado temporalmente en {temp_path}")

    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video_path = f"procesado_{file.filename}.mp4"
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    emotion_stats = {}
    frames_per_emotion = {}  # Guardar frames donde aparece cada emoción

    print(">>> Iniciando análisis completo del video...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\n>>> Fin del video.")
            break

        frame_count += 1
        faces = app_insight.get(frame)
        print(f">>> Frame {frame_count}/{total_frames} - Rostros: {len(faces)}")

        for i, face in enumerate(faces):
            x1, y1, x2, y2 = map(int, face.bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inputs = processor(images=img, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                emotion_id = probs.argmax().item()
                emotion_label = model.config.id2label[emotion_id]

            # Contabilizar
            emotion_stats[emotion_label] = emotion_stats.get(emotion_label, 0) + 1

            # Guardar primer frame donde aparece cada emoción
            if emotion_label not in frames_per_emotion:
                frames_per_emotion[emotion_label] = frame.copy()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion_label} ({probs[0][emotion_id]*100:.1f}%)",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

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

    # =====================================
    # Segunda emoción más frecuente (evitar neutral)
    # =====================================
    sorted_emotions = sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True)
    top_emotion = None
    for emo, count in sorted_emotions:
        if emo != 'neutral' and count > 0:
            top_emotion = emo
            porcentaje = (count / total_faces) * 100
            break
    if top_emotion is None:
        top_emotion = 'neutral'
        porcentaje = (emotion_stats['neutral'] / total_faces) * 100

    distribucion_porcentual = {
        e: round((count / total_faces) * 100, 2) for e, count in emotion_stats.items()
    }

    print(f"\n>>> Emoción seleccionada: {top_emotion.upper()} ({porcentaje:.2f}%)")
    print(">>> Distribución de emociones:", distribucion_porcentual)

    # =====================================
    # Imagen representativa de esa emoción
    # =====================================
    selected_frame = frames_per_emotion.get(top_emotion, None)
    image_base64 = None
    if selected_frame is not None:
        cv2.putText(selected_frame,
                    f"{top_emotion.upper()} ({porcentaje:.1f}%)",
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        output_image_path = f"resultado_{top_emotion}.jpg"
        cv2.imwrite(output_image_path, selected_frame)
        print(f"\n>>> Imagen representativa guardada: {output_image_path}")

        with open(output_image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        print(">>> Imagen convertida a base64 y lista para enviar.")
    else:
        print(">>> ⚠️ No se encontró frame claro de la emoción seleccionada.")

    # =====================================
    # Respuesta final
    # =====================================
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
