#  Lecture 10: Encoders and Decoders (Autoencoders)

##  What is an Autoencoder?

An **autoencoder** is a neural network trained to **reconstruct its input**.  
It learns a compressed internal representation known as the **latent code**.

The objective is not classification, but **representation learning**.

---

##  Autoencoder Architecture

An autoencoder consists of three main components:

1. **Encoder**
   - Maps the input data into a compressed representation

2. **Bottleneck (Latent Space)**
   - The most compact representation of the input
   - Forces the network to retain only essential features

3. **Decoder**
   - Reconstructs the original input from the latent code

---

##  Training Autoencoders

Training focuses on minimizing the **reconstruction error** between the input and the output.

### Key Hyperparameters

- **Code size (latent dimension)**
- **Number of layers**
- **Number of nodes per layer**
- **Reconstruction loss function**

---

##  Loss Functions in Autoencoders

Loss functions measure how close the reconstructed output is to the original input:

- **Mean Squared Error (MSE)**
  - Used for continuous-valued data

- **Binary Cross-Entropy (BCE)**
  - Used for binary or probability-based inputs

- **Kullback–Leibler (KL) Divergence**
  - Measures the difference between probability distributions

---

##  Types of Autoencoders

---

### 🔹 Undercomplete Autoencoders

- Latent dimension is **smaller** than input dimension  
- Forces compression  
- Prevents trivial identity mapping  

---

### 🔹 Overcomplete Autoencoders

- Latent dimension is **equal to or larger** than input dimension  
- Requires regularization to avoid simply copying the input  

---

##  Stochastic Encoders and Decoders

In stochastic autoencoders:

- The encoder outputs **parameters of a probability distribution**
- A latent vector is **sampled** from this distribution
- Introduces **uncertainty and randomness**
- Enables **smooth latent spaces**
- Forms the basis for **Variational Autoencoders (VAEs)**

---

##  Architecture of Stochastic Autoencoders

- **Stochastic Encoder**
  - Learns a distribution instead of a fixed latent vector

- **Stochastic Decoder**
  - Reconstructs input probabilistically from a sampled latent vector

---

##  Denoising Autoencoders

Denoising autoencoders are trained to:

- Reconstruct **clean input** from a **corrupted version**
- Improve robustness and feature learning

### Types of Noise

- Gaussian noise  
- Salt-and-pepper noise  
- Masking noise (randomly zeroing pixels)

---

##  Contractive Autoencoders

Contractive autoencoders aim to:

- Make representations **robust to small input changes**
- Penalize encoder sensitivity using an additional regularization term

### Loss Components

1. Reconstruction loss  
2. Regularization term (penalizes sensitivity of latent representation)

---

##  Comparison of Autoencoder Variants

| Variant | Main Idea | Goal | Strength |
|------|---------|------|---------|
| Basic Autoencoder | Reconstruct input | Compression | Simple and fast |
| Stochastic Encoder/Decoder | Output distributions | Uncertainty modeling | Basis for VAEs |
| Denoising Autoencoder | Remove noise | Robust feature learning | Learns essential features |
| Contractive Autoencoder | Penalize sensitivity | Smooth latent space | Strong regularization |

---

##  Applications of Autoencoders

- Image compression  
- Feature extraction  
- Image denoising  
- Anomaly and outlier detection  

---

##  References

- Khan et al., *Guide to CNNs for Computer Vision* (2018)  
- Chollet, *Deep Learning with Python* (2018)  
- Awad & Hassaballah, *Deep Learning in Computer Vision* (2020)  
- Elgendy, *Deep Learning for Vision Systems* (2020)
