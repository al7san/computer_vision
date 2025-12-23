# 🎬 Lecture 13: Learning on Videos, 3D Deep Learning, and Scene Graphs
## Overview

This lecture extends computer vision from **static images** to **videos and structured scenes**.
It focuses on understanding **relationships, motion, temporal dynamics, and activities** using
deep learning models.

---

## 🎯 Learning Objectives

By the end of this lecture, you should be able to:

- Explain **visual relationship modeling** between objects and scenes  
- Understand how **Graph Neural Networks (GNNs)** encode structured visual information  
- Describe **motion detection** and **multi-object tracking** pipelines  
- Infer **human actions and activities** from video sequences using temporal deep learning models  

---

## 🔗 Visual Relationships

**Visual relationships** describe how objects interact within a scene.

### Examples
- person **riding** horse  
- dog **next to** table  
- car **under** bridge  

They are represented as **triplets**:

(subject, predicate, object)

This representation captures both objects and their interactions.

---

## ❓ Why Visual Relationships Matter

Visual relationships enable **scene understanding**, not just object detection.

They support:
- Image captioning  
- Visual Question Answering (VQA)  
- Robotics reasoning  
- Surveillance and activity recognition  

They also allow **commonsense reasoning**, such as:
- “cup on table” implies physical support  

---

## 🧠 Modeling Visual Relationships

Typical pipeline:
1. **CNN-based object detection**
   - Extract object features
2. **Predicate classification**
   - Learn interactions between object pairs
3. **Graph-based reasoning**
   - Build **scene graphs**

### Key Idea
Combine:
- Appearance  
- Spatial layout  
- Context  

to model relationships accurately.

---

## 🕸️ Graph Neural Networks (GNNs) for Visual Reasoning

**Graph Neural Networks** process:
- **Nodes** → objects  
- **Edges** → relationships  

They are well suited for **scene graph representations**.

### Example Scene Graph
- Nodes: person, bike  
- Edge: riding  

---

## ❓ Why Use GNNs in Computer Vision?

GNNs:
- Encode structured relationships  
- Model contextual reasoning across objects  

They improve:
- Scene graph generation  
- Visual Question Answering  
- Relationship detection  
- Human–object interaction understanding  

---

## 🎞️ Motion Detection

**Motion detection** aims to identify pixels or regions that move across frames.

### Common Methods
- Frame differencing  
- Background subtraction  
- Optical flow (Horn–Schunck, Lucas–Kanade)  
- CNN-based motion segmentation  

---

## 🌊 Optical Flow

**Optical flow** estimates pixel-level motion vectors between consecutive frames.

### Applications
- Video stabilization  
- Action recognition  
- Object tracking  
- Autonomous driving  

---

## 🎯 Object Tracking

**Object tracking** maintains object identity across video frames.

### Types
- **Single-Object Tracking (SOT)**  
- **Multi-Object Tracking (MOT)**  

### Tracking Pipeline
1. **Detection**
   - CNN-based detectors (YOLO, Faster R-CNN)
2. **Tracking**
   - Kalman filters, SORT, DeepSORT
3. **Data Association**
   - Match detections to existing object tracks

---

## 🧍 Activity Recognition

**Activity recognition** identifies actions from image sequences or videos.

### Examples
- Running  
- Jumping  
- Cooking  
- Fighting  
- Playing sports  

---

## ⚠️ Challenges in Activity Recognition

- Temporal dependencies  
- Variations in viewpoint and scale  
- Occlusion  
- Multi-person interactions  
- Long and complex activities  

---

## 🛠️ Techniques for Activity Inference

### 1️⃣ CNN + LSTM Models
- CNN extracts frame-level features  
- LSTM / RNN models temporal sequences  

---

### 2️⃣ 3D CNNs
Examples:
- C3D  
- I3D  

- Perform convolution in **space and time**  
- Strong temporal modeling  

---

### 3️⃣ Transformers for Video
- Spatiotemporal attention  
- State-of-the-art approaches:
  - TimeSformer  
  - Video Swin Transformer  

---

### 4️⃣ Pose-Based Activity Recognition
- Track body keypoints (skeletons) over time  
- Effective for:
  - Sports analysis  
  - Gesture recognition  
  - Safety monitoring  

---

## 🧱 Example Architecture for Video Understanding

1. Input video frames  
2. CNN feature extraction (ResNet, EfficientNet)  
3. Sequence modeling (LSTM / GRU / Transformer)  
4. Activity classification layer  

---

## 🚀 Applications

- Video surveillance  
- Human action recognition  
- Autonomous driving  
- Robotics and human–robot interaction  
- Sports analytics  

---

## 📚 References

- Khan et al., *Guide to CNNs for Computer Vision* (2018)  
- Chollet, *Deep Learning with Python* (2018)  
- Awad & Hassaballah, *Deep Learning in Computer Vision* (2020)  
- Elgendy, *Deep Learning for Vision Systems* (2020)
