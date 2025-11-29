<div align="center">

# 🎭 Face Recognition with ArcFace

*Deep learning face recognition system powered by ResNet50 and DeiT with ArcFace loss*

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

- **ResNet50** - CNN-based approach with proven reliability
- **DeiT-Small** - Transformer-based architecture for modern ML

---

## ✨ Key Features

<table>
  <tr>
    <td>🔄</td>
    <td><b>Dual Architecture</b><br/>ResNet50 & DeiT support</td>
    <td>🎯</td>
    <td><b>High Accuracy</b><br/>80% validation accuracy</td>
  </tr>
  <tr>
    <td>👤</td>
    <td><b>Auto Detection</b><br/>MediaPipe face detection</td>
    <td>🌐</td>
    <td><b>Web Interface</b><br/>Interactive Gradio demo</td>
  </tr>
  <tr>
    <td>🚀</td>
    <td><b>Transfer Learning</b><br/>ImageNet pretrained</td>
    <td>⚡</td>
    <td><b>Real-time</b><br/>Fast inference pipeline</td>
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
│   ├── best_resnet50_arcface.pth
│   ├── best_deit_small_arcface.pth
│   └── label_map.json
│
├── 📂 train/
│   ├── train_resnet.ipynb      # ResNet50 training
│   └── train_deit.ipynb        # DeiT training
│
├── 📄 app.py                    # Gradio web interface
├── 📓 crop.ipynb
└── 📝 README.md
```

---

## 📊 Model Performance

<div align="center">

| 🏗️ Architecture | 📈 Val Accuracy
|:---------------:|:---------------:|
| **ResNet50** | 77% |
| **DeiT-Small** | 63% |

</div>

<table align="center">
  <tr>
    <td align="center"><b>ResNet50</b><br/><img src="train/output_resnet.png" alt="ResNet50 Training Curve" width="60%"/></td>
    <td align="center"><b>DeiT-Small</b><br/><img src="train/output_deit.png" alt="DeiT-Small Training Curve" width="62%"/></td>
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
jupyter notebook train/train_resnet.ipynb    # For ResNet50
jupyter notebook train/train_deit.ipynb      # For DeiT
```

**🔮 Inference**
```bash
# Run local Gradio interface
python app.py
# Access at http://localhost:7860
```

**🌐 Try Online**

No installation needed! Try our live demo:

<div align="center">

**[🤗 Launch Live Demo](https://huggingface.co/spaces/ditorifki/face-recognition-demo)**

</div>

---

## ⚙️ Technical Details

### Architecture Configuration

```yaml
Image Size: 224 × 224
Embedding Dimension: 512
ArcFace Scale (s): 25.0
ArcFace Margin (m): 0.10
Optimizer: Adam (lr=1e-4)
Scheduler: CosineAnnealingLR
```

### Dataset Statistics

```yaml
Total Classes: 70 identities
Total Images: 283
Training Split: 213 images (80%)
Validation Split: 70 images (20%)
Augmentation: Flip, Affine, Color Jitter
```

---

## 🧮 ArcFace Loss Function

ArcFace enhances face discrimination by adding an angular margin to the cosine similarity:

```
L = -log(e^(s·cos(θ+m)) / (e^(s·cos(θ+m)) + Σe^(s·cos(θ))))
```

**Where:**
- `s` = scale parameter (25.0)
- `m` = angular margin (0.10)
- `θ` = angle between feature and weight vectors

**Benefits:**
- ✅ Enhanced intra-class compactness
- ✅ Improved inter-class separability
- ✅ Better generalization to unseen faces

---

## 🛠️ Built With

<div align="center">

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

**Core Libraries:**
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
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