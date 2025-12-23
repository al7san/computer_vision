# 🎯 Lecture 11: Detection, Segmentation, and Feature Extraction in Computer Vision

## ❓ Why Detection & Segmentation?

Detection and segmentation are essential for many real-world computer vision tasks, including:

- Autonomous driving  
- Medical imaging  
- Robotics and object tracking  
- Surveillance systems  
- Scene understanding  

These tasks require not only recognizing objects, but also understanding **where** they are and **how** they occupy space in the image.

---

## 🧩 Semantic Segmentation

**Semantic segmentation** assigns a **class label to every pixel** in an image.

- All objects belonging to the same class share the same label  
- No distinction between individual object instances  

### Common Architectures

- **Fully Convolutional Networks (FCN)**  
- **U-Net** (uses skip connections)  
- **SegNet**  
- **DeepLab** (v1 – v3+)  

Semantic segmentation is widely used in applications such as road scene understanding and medical image analysis.

---

## 📦 Object Detection

**Object detection** locates and classifies **each object instance** in an image using a **bounding box**.

### Two Main Components

- **Classification**: What is the object?  
- **Localization**: Where is the object?  

The output for each detected object includes:
- Bounding box coordinates (x, y, width, height)  
- Class label  
- Confidence score  

---

## 🔁 Object Detection Pipeline

### 🔹 Two-Stage Detectors

Examples:
- R-CNN  
- Fast R-CNN  
- Faster R-CNN  

**Process:**
1. Generate region proposals  
2. Perform classification and bounding box regression  

- High accuracy  
- Slower inference  

---

### 🔹 One-Stage Detectors

Examples:
- YOLO (You Only Look Once)  
- SSD (Single Shot Detector)  
- RetinaNet  

**Characteristics:**
- No region proposal stage  
- Faster and suitable for real-time detection  

---

## 🎭 Instance Segmentation

**Instance segmentation** performs **pixel-level segmentation for each object instance separately**.

It combines:
- Object detection  
- Semantic segmentation  

### How It Works

1. Detect each object using a bounding box  
2. Generate a pixel-wise mask for each object inside the box  

This allows distinguishing between different objects of the same class.

---

## 🧠 Feature Extraction in Classical Computer Vision

Before deep learning, computer vision relied on **hand-crafted features**.

---

### 🔹 Histogram of Oriented Gradients (HoG)

HoG extracts **edge orientation and shape information**.

#### Steps:
1. Compute image gradients  
2. Create orientation histograms in small cells  
3. Normalize over blocks  
4. Concatenate into a feature vector  

#### Strengths:
- Effective for shape detection (e.g., pedestrians)  
- Robust to illumination and contrast changes  

---

### 🔹 Scale-Invariant Feature Transform (SIFT)

SIFT detects and describes **stable keypoints** in images.

#### Key Steps:
- Detect scale-space extrema  
- Localize keypoints  
- Assign orientations  
- Generate descriptors  

SIFT is invariant to scale and rotation, making it useful for matching tasks.

---

## ⚖️ Classical vs Deep Features

| Aspect | Classical Features (HoG / SIFT) | Deep Features (CNN) |
|-----|-------------------------------|---------------------|
| Feature Design | Manual | Learned from data |
| Robustness | Moderate | Very high |
| Performance | Good | State-of-the-art |
| Data Requirement | Low | High |

---

## 🚀 Applications

- Autonomous driving  
- Medical image analysis  
- Surveillance systems  
- Robotics and tracking  
- Scene understanding  

---

## 📚 References

- Khan et al., *Guide to CNNs for Computer Vision* (2018)  
- Chollet, *Deep Learning with Python* (2018)  
- Awad & Hassaballah, *Deep Learning in Computer Vision* (2020)  
- Elgendy, *Deep Learning for Vision Systems* (2020)
