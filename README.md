<div align="center">

# 🎭 Face Recognition with ArcFace

*Deep learning face recognition system powered by FaceNet (InceptionResnetV1) with ArcFace loss*

[![Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/ditorifki/face-recognition-demo)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

</div>

---

## 👥 Team

<table align="center">
  <tr>
    <td align="center"><b>Fathan Andi Kartagama</b><br/>122140055</td>
    <td align="center"><b>Rahmat Aldi Nasda</b><br/>122140077</td>
    <td align="center"><b>Dito Rifki Irawan</b><br/>122140153</td>
  </tr>
</table>

---

## 📖 Overview

This project implements a state-of-the-art face recognition system using **ArcFace loss** to learn highly discriminative facial embeddings. The system supports two powerful backbone architectures:

- **FaceNet (InceptionResnetV1)** - Pretrained on VGGFace2 for robust face recognition
- **DeiT-Small** - Transformer-based architecture for comparison

---

## ✨ Key Features

<table>
  <tr>
    <td>🧠</td>
    <td><b>Dual Architecture</b><br/>FaceNet & DeiT support</td>
    <td>🎯</td>
    <td><b>High Accuracy</b><br/>99% validation accuracy</td>
  </tr>
  <tr>
    <td>👤</td>
    <td><b>Auto Detection</b><br/>MediaPipe face detection</td>
    <td>🌐</td>
    <td><b>Web Interface</b><br/>Interactive Gradio demo</td>
  </tr>
  <tr>
    <td>🚀</td>
    <td><b>Transfer Learning</b><br/>VGGFace2 & ImageNet pretrained</td>
    <td>⚡</td>
    <td><b>Real-time</b><br/>Fast inference pipeline</td>
  </tr>
  <tr>
    <td>📊</td>
    <td><b>Attendance System</b><br/>Automated logging with timestamps</td>
    <td>🔄</td>
    <td><b>K-Fold Training</b><br/>5-fold cross validation</td>
  </tr>
</table>

---

## 📁 Project Structure

```
face-recognition/
│
├── 📂 data/
│   └── Train_Cropped/          # 70 identity classes cropped via crop.ipynb
│   └── Train/ 
│
├── 📂 models/
│   ├── best_facenet_arcface_kfold5.pth
│   ├── best_deit_small_patch16_224.fb_in1k_arcface_kfold5.pth
│   └── label_map.json
│
├── 📂 train/
│   ├── train_facenet.ipynb     # FaceNet training with K-Fold
│   └── train_deit.ipynb        # DeiT training with K-Fold
│
├── 📄 app.py                    # Gradio web interface with attendance
├── 📄 attendance_log.csv        # Attendance records
├── 📓 crop.ipynb
├── 📓 evaluation.ipynb           # Evaluate trained face recognition model on test dataset
└── 📝 README.md
```

---

## 📊 Model Performance

<div align="center">

| 🏗️ Architecture | 📈 Val Accuracy | 🎯 Pretrained |
|:---------------:|:---------------:|:----------------:|
| **InceptionResnetV1 + ArcFace** | 99% | VGGFace2 |
| **DeiT-Small + ArcFace** | 81% | ImageNet-1k |

</div>

<table align="center">
  <tr>
    <td align="center"><b>FaceNet (InceptionResnetV1)</b><br/><img src="train/output_facenet.png" alt="FaceNet Training Curve" width="60%"/></td>
    <td align="center"><b>DeiT-Small</b><br/><img src="train/output_deit.png" alt="DeiT-Small Training Curve" width="68%"/></td>
  </tr>
</table>

---

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/Caseinn/face-recognition.git
cd face-recognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**🎓 Training**
```bash
# Launch Jupyter and open training notebooks
jupyter notebook train/train_facenet.ipynb   # For FaceNet
jupyter notebook train/train_deit.ipynb      # For DeiT
```

**🔮 Inference & Attendance**
```bash
# Run local Gradio interface
python app.py
# Access at http://localhost:7860
```

**🌐 Try Online**

No installation needed! Try our live demo:

<div align="center">

**[🤗 Launch Live Demo](https://huggingface.co/spaces/ditorifki/attendance-system)**

</div>

---

## ⚙️ Technical Details

### Architecture Configuration

**FaceNet (InceptionResnetV1) - Main Model:**
```yaml
Backbone: InceptionResnetV1 (VGGFace2 pretrained)
Input Size: 160 × 160
Embedding Dimension: 512
ArcFace Scale (s): 25.0
ArcFace Margin (m): 0.30
Optimizer: Adam (lr=1e-4)
Scheduler: CosineAnnealingLR
Training Strategy: 5-Fold Cross Validation
```

**DeiT-Small - Comparison Model:**
```yaml
Backbone: DeiT-Small (ImageNet pretrained)
Input Size: 224 × 224
Embedding Dimension: 512
ArcFace Scale (s): 25.0
ArcFace Margin (m): 0.30
Optimizer: Adam (lr=1e-4)
Scheduler: CosineAnnealingLR
Training Strategy: 5-Fold Cross Validation
```

### Dataset Statistics

```yaml
Total Classes: 70
Total Images: 283
Face Detection: MediaPipe (dual-pass detection)
Face Crop:
  Size: 384x384
  Margin: 15%
Augmentation:
  - Random Horizontal Flip: 50%
  - Random Affine Transform: 60%
    - Rotation: ±20°
    - Translation: ±10%
    - Scale: 0.85–1.15
  - Random Perspective: 30%
    - Distortion Scale: 0.2
  - Gaussian Blur: [20%, 40%]
  - Color Jitter:
      Brightness: ±30%
      Contrast: ±30%
      Saturation: ±30%
      Hue: ±10%
  - Random Grayscale: 15%
  - Motion Blur: 20%
  - Gaussian Noise: 40%
    - Std: 0.05
  - ISO Noise: 40%
  - Multiplicative Noise: 40%
  - Brightness Enhancement: 30% 
  - Random Shadow: 20%
  - JPEG Compression: 20%
    - Quality: 40–80
  - Random Erasing: 30%
    - Area: 2%–15% 
```

---

## 🧮 ArcFace Loss Function

ArcFace enhances face discrimination by adding an angular margin to the cosine similarity:

```
L = -log(e^(s·cos(θ+m)) / (e^(s·cos(θ+m)) + Σe^(s·cos(θ))))
```

**Where:**
- `s` = scale parameter (25.0)
- `m` = angular margin (0.30)
- `θ` = angle between feature and weight vectors

**Benefits:**
- ✅ Enhanced intra-class compactness
- ✅ Improved inter-class separability
- ✅ Better generalization to unseen faces
- ✅ State-of-the-art performance on face verification

---

## 📋 Attendance System

The system includes an automated attendance logging feature:

**Features:**
- ✅ Real-time face recognition
- ✅ Automatic timestamp recording
- ✅ Confidence score logging
- ✅ CSV export for records
- ✅ Live attendance log viewer

**Log Format:**
```csv
Timestamp,Name,Confidence,Status
2024-12-01 14:30:45,John Doe,0.9523,Success
```

---

## 🛠️ Built With

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

**Core Libraries:**
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) - FaceNet implementation
- [timm](https://github.com/huggingface/pytorch-image-models) - DeiT & Vision Transformers
- [MediaPipe](https://mediapipe.dev/) - Face Detection
- [Gradio](https://gradio.app/) - Web Interface
- [ArcFace](https://arxiv.org/abs/1801.07698) - Loss Function Implementation

---

## 📚 Course Information

<div align="center">

**Deep Learning (Pembelajaran Mendalam)**  
Semester 7 | Final Project

</div>

---

<div align="center">

*[⭐ Star this repo](https://github.com/Caseinn/face-recognition) if you find it helpful!*

</div>
