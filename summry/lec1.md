
# 🧠 Lecture 1: Introduction to Computer Vision & Image Formation
**Course:** CIS 6217 – Computer Vision for Data Representation  
**Institution:** College of Computer Science, King Khalid University  

---

## 🎯 Learning Outcomes
By the end of this lecture, you should be able to:
- Define **computer vision (CV)** and explain its importance.
- Describe the **historical evolution** of computer vision.
- Explain the **basics of image formation** (geometry and photometry).
- Understand **pixels, color spaces, and image representation**.

---

## 👁️ What is Computer Vision?
Computer Vision (CV) is a field of **Artificial Intelligence (AI)** that enables machines to **see, interpret, and understand** the visual world.

### Key Ideas
- Mimics **human visual perception** using algorithms and models.  
- Bridges the gap between **raw image data** and **decision-making**.

### In short:
> 🧩 Computer vision = *Algorithms that make sense of visual data.*

---

## 🚗 Applications of Computer Vision

| Field | Example Applications |
|-------|----------------------|
| **Autonomous Vehicles** | Lane detection, obstacle recognition |
| **Healthcare** | Tumor detection, X-ray analysis |
| **Security** | Face recognition, biometrics |
| **Augmented/Virtual Reality** | Object tracking, gesture recognition |
| **Industrial Automation** | Quality inspection, robot vision |

> 💡 If a machine needs to "see" or "analyze" visuals — it uses Computer Vision.

---

## 🕰️ Historical Development of Computer Vision

| Era | Key Characteristics |
|-----|----------------------|
| **1960s–1980s** | Early research in pattern recognition and edge detection |
| **1990s–2000s** | Classical CV: SIFT, HoG, image segmentation |
| **2012–Present** | Deep Learning revolution (CNNs, GANs, Transformers) |

> 2012 (AlexNet) marked the beginning of modern deep-learning-based vision.

---

## 🌅 Image Formation
**Forsyth & Ponce (2010)** define image formation as:  
> “Geometry tells us where the light goes; radiometry tells us how much arrives.”

So, image formation involves:
1. **Geometry** → how 3D scenes project onto 2D.  
2. **Photometry** → how light interacts with surfaces.

---

## 📐 Image Formation: Geometry — *Pinhole Camera Model*

> Geometry explains how 3D scenes are projected onto a 2D image plane.

```mermaid
graph TD
A[3D Object] --> B((Light Rays))
B --> C((Pinhole))
C --> D[Inverted Image on 2D Plane]
D --> E[Camera Sensor / Image Plane]
````

### Key Parameters

* **Focal length (f)** – distance from pinhole to image plane
* **Aperture size** – controls brightness and depth of field
* **Field of view (FOV)** – extent of the observable scene

> The result: a **perspective projection** where parallel lines appear to converge (vanishing point).

---

## 💡 Image Formation: Photometry

**Photometry** studies how light intensity affects the appearance of images.

### Light Interaction with Objects

* **Reflection:** Light bounces off surfaces
* **Absorption:** Light energy absorbed by material
* **Transmission:** Light passes through transparent objects

### Factors Influencing Image Brightness

* Illumination conditions
* Surface reflectance properties
* Sensor exposure and gain

📊 *Radiometry* measures energy per unit area per solid angle —
the physical foundation behind **brightness**, **contrast**, and **intensity**.

---

## 🖼️ Digital Image

A **digital image** is a 2D array of small elements called **pixels**,
each pixel holding intensity or color information.

---

## 🧩 Digital Image Representation

| Concept          | Description                                           |
| ---------------- | ----------------------------------------------------- |
| **Pixel**        | The smallest image unit                               |
| **Resolution**   | Number of pixels along width × height                 |
| **Aspect Ratio** | Proportion between width and height                   |
| **Color Depth**  | Number of bits per pixel (usually 8 bits per channel) |

### Example

* **Grayscale:** 2D matrix (H × W)
* **Color (RGB):** 3D array (H × W × 3)

> Example: A 256×256 RGB image = 256 × 256 × 3 ≈ 200,000 pixel values.

---

## 🌈 Intensity & Color Representation — *RGB Model*

> The RGB model combines Red, Green, and Blue light to produce colors.

```mermaid
graph TD
R[Red Channel] --> C[Combined RGB Color]
G[Green Channel] --> C
B[Blue Channel] --> C
C --> O[Perceived Color on Screen]
```

| Type            | Description                                  |
| --------------- | -------------------------------------------- |
| **Grayscale**   | One value per pixel (0 = black, 255 = white) |
| **Color (RGB)** | Each pixel = [R, G, B] values (0–255)        |

**Examples**

* (255, 0, 0) → Red
* (0, 255, 0) → Green
* (0, 0, 255) → Blue
* (255, 255, 255) → White
* (0, 0, 0) → Black

---

## 🎨 Common Color Models

| Model     | Components                          | Use Case                         |
| --------- | ----------------------------------- | -------------------------------- |
| **RGB**   | Red, Green, Blue                    | Displays, image sensors          |
| **HSV**   | Hue, Saturation, Value              | Color segmentation and filtering |
| **YCbCr** | Luminance (Y), Chrominance (Cb, Cr) | Video compression (JPEG, MPEG)   |

> Different color models are used for different tasks — e.g., HSV for color-based segmentation.

---

## 🧮 Digital Image as a Matrix

An image is stored as a **matrix (array)** in memory.

| Image Type      | Representation  |
| --------------- | --------------- |
| **Grayscale**   | H × W matrix    |
| **Color (RGB)** | H × W × 3 array |

Example:

```python
import cv2
img = cv2.imread("image.jpg")
print(img.shape)  # (Height, Width, 3)
```

> Each pixel is an element containing intensity or color channel values.

---

## 🧰 Image Data in Computer Vision

Images are represented as **multidimensional arrays** (height × width × channels).

### Typical CV Libraries

* **OpenCV (cv2)** – most widely used for computer vision tasks
* **Pillow (PIL)** – simple image loading and manipulation
* **TensorFlow / PyTorch** – deep learning frameworks

### Common Operations

* Load, display, and resize images
* Convert between color spaces (e.g., RGB → Grayscale)
* Access or modify pixel values

---

## 🐍 Python Example: Image Handling

```python
import cv2

# Load an image
img = cv2.imread("cat.jpg")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display
cv2.imshow("Grayscale Image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

**Explanation**

* `cv2.imread()` loads the image as a NumPy array.
* `cv2.cvtColor()` converts color space (here BGR → Grayscale).
* `imshow()` displays the processed image.

---

## 📚 Reference

Forsyth, D., & Ponce, J. (2010). *Computer Vision: A Modern Approach.*
Pearson Education.

---

## ✅ Summary Checklist

| Concept              | You Should Be Able To…                       |
| -------------------- | -------------------------------------------- |
| Computer Vision      | Define CV and list real-world applications   |
| Image Formation      | Explain geometry and photometry              |
| Image Representation | Describe pixels, resolution, and color depth |
| Color Models         | Differentiate between RGB, HSV, and YCbCr    |
| Image Matrix         | Understand how images are stored in memory   |
| Python Practice      | Load and manipulate images using OpenCV      |

---

## 🧠 Final Note

This lecture builds the **foundation for understanding all future CV topics** —
such as **camera calibration**, **object detection**, and **image segmentation**.

> “Once you understand how an image is formed,
> you understand how a computer can begin to see.” 👁️
