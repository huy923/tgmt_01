# 🎭 AI Face Analysis Suite

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.12.0-green?style=for-the-badge&logo=opencv&logoColor=white)
![DeepFace](https://img.shields.io/badge/DeepFace-0.0.95-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red?style=for-the-badge&logo=pytorch&logoColor=white)

**A powerful real-time face analysis system that detects emotions, predicts age, recognizes faces, and more!**

[🚀 Quick Start](#-quick-start) • [📋 Features](#-features) • [🛠️ Installation](#️-installation) • [📖 Usage](#-usage) • [🎯 Examples](#-examples)

</div>

---

## ✨ Features

### 🎥 **Real-Time Analysis**
- **Live Webcam Processing** - Real-time face detection and analysis
- **Multi-Attribute Detection** - Emotion, gender, and age prediction simultaneously
- **GPU Acceleration** - Automatic GPU detection and optimization for faster processing
- **Performance Optimized** - Smart frame skipping and caching for smooth performance

### 🧠 **Advanced AI Models**
- **DeepFace Integration** - State-of-the-art emotion and gender recognition
- **OpenCV DNN** - High-accuracy age prediction using Caffe models
- **Haar Cascade Detection** - Robust face detection with multiple algorithms
- **LBPH Face Recognition** - Train custom face recognition models

### 🎨 **Visual Interface**
- **Real-time Bounding Boxes** - Clean, colorful face detection overlays
- **Live Data Display** - Real-time emotion, age, and gender information
- **Customizable Camera** - Support for internal and external webcams
- **Responsive Design** - Optimized for different screen sizes

### 📊 **Data Management**
- **Dataset Collection** - Automated face image capture for training
- **Model Training** - Custom face recognition model creation
- **Batch Processing** - Process multiple images at once
- **Export Capabilities** - Save results and trained models

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- CUDA-compatible GPU (optional, for acceleration)

### Quick Setup

<details>
<summary><strong>🐍 Virtual Environment (Recommended)</strong></summary>

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>🐍 Conda Environment</strong></summary>

```bash
# Create conda environment
conda create -n face-analysis python=3.8

# Activate environment
conda activate face-analysis

# Install dependencies
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>📦 Direct Installation</strong></summary>

```bash
# Install required packages
pip install opencv-python==4.12.0.88
pip install opencv-contrib-python==4.12.0.88
pip install deepface==0.0.95
pip install tf-keras==2.20.1
pip install torch==2.8.0
```

</details>

---

## 🚀 Quick Start

### 1. **Real-Time Face Analysis**
```bash
python main.py
```
*Opens webcam and displays live emotion, age, and gender analysis*

### 2. **Image Age Prediction**
```bash
python ShowAge.py
```
*Processes images from the dataset folder and predicts ages*

### 3. **Face Recognition Training**
```bash
python FaceRecognition.py
```
*Captures training images and trains a face recognition model*

---

## 📖 Usage

### 🎥 Real-Time Analysis (`main.py`)

The main application provides live face analysis with the following features:

```python
# Camera Configuration
cam = cv2.VideoCapture(0)  # 0 for laptop webcam, 2 for external webcam

# Performance Settings
analysis_skip_frames = 5    # Analyze every 5th frame
analysis_interval = 0.5     # Analyze every 0.5 seconds
```

**Key Features:**
- **Smart Performance** - Analyzes every 5th frame to maintain smooth FPS
- **GPU Detection** - Automatically detects and uses available GPU acceleration
- **Error Handling** - Robust error handling for analysis failures
- **Live Display** - Real-time emotion, gender, and age information

### 🖼️ Image Processing (`ShowAge.py`)

Process static images for age prediction:

```python
# Process a single image
image_path = "./dataset/huy_0.jpg"
process_image(image_path)
```

**Features:**
- **Batch Processing** - Process multiple images from dataset
- **High Accuracy** - Uses OpenCV DNN with Caffe models
- **Visual Results** - Displays results with bounding boxes and labels

### 👤 Face Recognition (`FaceRecognition.py`)

Train custom face recognition models:

```python
# Capture training images
capture_images("username")

# Train model
label = {"username": 0, "another_user": 1}
recognizer = train_model(label)
```

**Workflow:**
1. **Capture** - Collect 100 face images per person
2. **Train** - Create LBPH face recognition model
3. **Predict** - Use trained model for face identification

---

## 🎯 Examples

### Real-Time Analysis Output
```
🎭 Detected Face:
   👤 Gender: Male
   🎂 Age: (25-32)
   😊 Emotion: Happy
   📊 Confidence: 95%
```

### Supported Emotions
- 😊 Happy
- 😢 Sad
- 😠 Angry
- 😨 Fear
- 😮 Surprise
- 😑 Neutral
- 😖 Disgust

### Age Ranges
- 👶 (0-2) - Baby
- 🧒 (3-6) - Toddler
- 👦 (7-12) - Child
- 👨 (13-17) - Teen
- 👨‍💼 (18-24) - Young Adult
- 👨‍💻 (25-32) - Adult
- 👨‍🏫 (33-39) - Mature Adult
- 👨‍🔬 (40-45) - Middle-aged
- 👨‍🎓 (46-50) - Senior Adult
- 👴 (51+) - Elderly

---

## ⚙️ Configuration

### Camera Settings
```python
# In main.py, adjust camera source:
cam = cv2.VideoCapture(0)  # Laptop webcam
cam = cv2.VideoCapture(2)  # External webcam
```

### Performance Tuning
```python
# Adjust analysis frequency
analysis_skip_frames = 5    # Higher = better performance
analysis_interval = 0.5     # Lower = more frequent analysis
```

### GPU Configuration
The system automatically detects and configures GPU acceleration:
- **CUDA Support** - Automatic detection and configuration
- **OpenCV DNN** - GPU-accelerated age prediction
- **PyTorch** - GPU-accelerated emotion analysis

---

## 📁 Project Structure

```
📦 AI Face Analysis Suite
├── 🎥 main.py                 # Real-time face analysis
├── 🖼️ ShowAge.py              # Image age prediction
├── 👤 FaceRecognition.py      # Face recognition training
├── 🧪 test.py                 # Testing utilities
├── 📋 requirements.txt        # Dependencies
├── 📁 model/                  # AI models
│   ├── age_deploy.prototxt    # Age prediction config
│   ├── age_net.caffemodel     # Age prediction weights
│   ├── opencv_face_detector.pbtxt
│   └── opencv_face_detector_uint8.pb
├── 📁 dataset/                # Training images
│   └── *.jpg                  # Face training data
└── 📄 README.md               # This file
```

---

## 🔧 Troubleshooting

### Common Issues

<details>
<summary><strong>🚫 Camera not working</strong></summary>

```bash
# Check camera permissions
ls /dev/video*

# Test camera with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

</details>

<details>
<summary><strong>🐌 Slow performance</strong></summary>

```python
# Increase skip frames for better performance
analysis_skip_frames = 10  # Analyze every 10th frame
analysis_interval = 1.0    # Analyze every 1 second
```

</details>

<details>
<summary><strong>💾 Memory issues</strong></summary>

```python
# Reduce image resolution
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
```

</details>

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **🐛 Report Bugs** - Open an issue with detailed information
2. **💡 Suggest Features** - Propose new functionality
3. **🔧 Submit PRs** - Contribute code improvements
4. **📚 Improve Docs** - Help with documentation

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/ai-face-analysis-suite.git
cd ai-face-analysis-suite

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python test.py
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OpenCV** - Computer vision library
- **DeepFace** - Emotion and gender recognition
- **PyTorch** - Deep learning framework
- **TensorFlow** - Machine learning platform

---

## 📞 Support

Need help? Here are some resources:

- 📧 **Email**: [your-email@example.com](mailto:your-email@example.com)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/ai-face-analysis-suite/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-face-analysis-suite/discussions)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and AI

[⬆️ Back to Top](#-ai-face-analysis-suite)

</div>