# ============================================
# BLOCK 1 – IMPORTS & BASIC CONFIG
# ============================================
import json
import numpy as np
from typing import Optional

from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import mediapipe as mp
import gradio as gr

# ---- Model & data config ----
MODEL_PATH = "models/best_resnet50_arcface.pth"
LABEL_MAP_PATH = "models/label_map.json"
IMG_SIZE = 224  # input size model

# ---- Face crop config (mirip script crop-mu) ----
TARGET_SIZE = (384, 384)   # ukuran crop akhir sebelum di-resize ke 224
MARGIN_RATIO = 0.15
MAX_DIM = 1600
MIN_DIM = 256
USE_CENTER_FALLBACK = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================
# BLOCK 2 – LOAD LABEL MAP
# ============================================
with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# label_map: {"NamaOrang": idx}
idx_to_name = {v: k for k, v in label_map.items()}


# ============================================
# BLOCK 3 – MEDIAPIPE FACE DETECTION HELPERS
# ============================================
mp_face_detection = mp.solutions.face_detection


def run_face_detection(img_rgb: np.ndarray):
    """
    Jalankan MediaPipe FaceDetection:
    - Pertama model_selection=1 (full range)
    - Kalau gagal, fallback ke model_selection=0 (short range) dengan threshold lebih longgar.
    """
    h, w = img_rgb.shape[:2]

    # Pass 1: full range
    with mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    ) as face_det:
        res = face_det.process(img_rgb)

    if res.detections:
        return res, h, w

    # Pass 2: short range sebagai fallback
    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.3
    ) as face_det:
        res2 = face_det.process(img_rgb)

    return res2, h, w


def pick_best_detection(detections, img_w: int, img_h: int):
    """
    Pilih wajah terbaik berdasarkan score * area relatif.
    Mirip heuristik 'wajah paling besar & paling yakin'.
    """
    best = None
    best_score = -1.0
    for det in detections:
        bbox = det.location_data.relative_bounding_box
        score = det.score[0]
        area = max(bbox.width * bbox.height, 1e-6)
        combined = score * area
        if combined > best_score:
            best_score = combined
            best = det
    return best


def detect_and_crop_from_pil(pil_img: Image.Image) -> Optional[Image.Image]:
    """
    Mirip fungsi detect_and_crop(image_path) versi kamu,
    tapi diadaptasi untuk input PIL dan output PIL.
    """
    # Fix orientasi EXIF dan pastikan RGB
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    img = np.array(pil_img)  # RGB
    orig_img = img.copy()
    orig_h, orig_w = orig_img.shape[:2]

    # Scale up bila terlalu kecil
    short_side = min(orig_h, orig_w)
    if short_side < MIN_DIM:
        scale = MIN_DIM / float(short_side)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pil_img_resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
        img = np.array(pil_img_resized)

    # Scale down bila terlalu besar
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img_resized = pil_img.resize((new_w, new_h), Image.BICUBIC)
        img = np.array(pil_img_resized)
        h, w = new_h, new_w

    # Deteksi di gambar yang sudah di-scale
    results, h, w = run_face_detection(img)

    # Kalau belum dapat apa-apa, coba di resolusi asli
    if not results or not results.detections:
        img = orig_img
        h, w = img.shape[:2]
        results, h, w = run_face_detection(img)

    # Kalau tetap gagal → fallback / None
    if not results or not results.detections:
        if not USE_CENTER_FALLBACK:
            return None

        # Center crop fallback (80% dari sisi terpendek)
        side = int(0.8 * min(h, w))
        cx, cy = w // 2, h // 2
        x1 = max(cx - side // 2, 0)
        y1 = max(cy - side // 2, 0)
        x2 = min(cx + side // 2, w)
        y2 = min(cy + side // 2, h)
        cropped = img[y1:y2, x1:x2]
        face_pil = Image.fromarray(cropped)
        face_pil = ImageOps.fit(face_pil, TARGET_SIZE, Image.BICUBIC)
        return face_pil

    # Pilih detection terbaik
    best_det = pick_best_detection(results.detections, w, h)
    bbox = best_det.location_data.relative_bounding_box

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    x = max(0, x)
    y = max(0, y)
    bw = max(1, bw)
    bh = max(1, bh)

    # Tambah margin
    margin_x = int(bw * MARGIN_RATIO)
    margin_y = int(bh * MARGIN_RATIO)

    x1 = max(x - margin_x, 0)
    y1 = max(y - margin_y, 0)
    x2 = min(x + bw + margin_x, w)
    y2 = min(y + bh + margin_y, h)

    crop = img[y1:y2, x1:x2]
    face_pil = Image.fromarray(crop)
    face_pil = ImageOps.fit(face_pil, TARGET_SIZE, Image.BICUBIC)
    return face_pil


# ============================================
# BLOCK 4 – ARCFAce MODEL (ResNet50)
# ============================================
class ArcMarginProduct(nn.Module):
    """
    Sama seperti yang dipakai waktu training (harus konsisten).
    Di inference kita cuma pakai weight-nya & plain cosine.
    """
    def __init__(self, in_features, out_features, s=25.0, m=0.10, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = np.cos(m)
        self.sin_m = np.sin(m)
        self.th = np.cos(np.pi - m)
        self.mm = np.sin(np.pi - m) * m

    def forward(self, input, label):
        # Dipakai hanya saat TRAINING (waktu dulu melatih model)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.clamp(cosine.pow(2), 0.0, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ResNet50ArcFace(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, s=25.0, m=0.10):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        # freeze tidak wajib buat inference, tapi aman
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.arc_margin = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=s,
            m=m,
            easy_margin=False,
        )

    def forward(self, x):
        feat = self.backbone(x)               # (B, 2048)
        emb = self.embedding(feat)            # (B, 512)
        emb = F.normalize(emb, dim=1)

        # Inference: plain cosine logits (tanpa margin),
        # tapi tetap pakai weight ArcFace
        logits = F.linear(
            F.normalize(emb),
            F.normalize(self.arc_margin.weight)
        ) * self.arc_margin.s

        return logits, emb


# ============================================
# BLOCK 5 – LOAD MODEL & PREPROCESS
# ============================================
def load_model():
    num_classes = len(label_map)
    model = ResNet50ArcFace(
        num_classes=num_classes,
        embedding_dim=512,
        s=25.0,
        m=0.20,
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


model = load_model()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def prepare_image(pil_img: Image.Image) -> torch.Tensor:
    pil_img = pil_img.convert("RGB")
    img = preprocess(pil_img)
    img = img.unsqueeze(0)
    return img.to(device)


# ============================================
# BLOCK 6 – INFERENCE PIPELINE
# ============================================
def predict_with_crop(img: Image.Image):
    """
    Fungsi utama untuk Gradio:
    - input: PIL image (raw upload)
    - output: (cropped_face_PIL, dict label->prob)
    """
    if img is None:
        return None, {"No image": 1.0}

    # 1) Crop wajah dulu
    face_pil = detect_and_crop_from_pil(img)
    if face_pil is None:
        return None, {"No face detected": 1.0}

    # 2) Preprocess & inference
    x = prepare_image(face_pil)
    with torch.no_grad():
        logits, emb = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    probs_np = probs.cpu().numpy()
    out_dict = {idx_to_name[i]: float(probs_np[i]) for i in range(len(idx_to_name))}

    return face_pil, out_dict


# ============================================
# BLOCK 7 – GRADIO UI
# ============================================
with gr.Blocks() as demo:
    gr.Markdown("<h1 id='title'>Face Recognition Demo</h1>")
    gr.Markdown("<p id='description'>Upload a photo with a face. The system will detect and crop the face using MediaPipe, then predict identity with ResNet50 + ArcFace.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Photo",
                height=600
            )
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary", size="lg", scale=1)
                predict_btn = gr.Button("Analyze Face", variant="primary", size="lg", scale=2)
                
        with gr.Column(scale=1):
            cropped_face = gr.Image(
                type="pil",
                label="Detected Face",
                height=302
            )
            prediction = gr.Label(
                num_top_classes=5,
                label="Identity Prediction (Top-5)"
            )
    
    predict_btn.click(
        fn=predict_with_crop,
        inputs=input_image,
        outputs=[cropped_face, prediction]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, None),
        inputs=None,
        outputs=[input_image, cropped_face, prediction]
    )

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto;
            }
            #title {
                text-align: center;
                font-size: 2.5em;
                font-weight: 700;
                margin-bottom: 0.5em;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            #description {
                text-align: center;
                color: #666;
                font-size: 1.1em;
                margin-bottom: 2em;
            }
            .image-container img {
                object-fit: contain !important;
            }
        """
    )
