# 📘 CIS 6217 – Comprehensive Review Notes (Lectures 1–13)

---

## Lecture 1 – Introduction to Computer Vision
**Covered topics**
- Definition and goals of Computer Vision
- Human Vision vs Computer Vision
- Core CV tasks: classification, detection, segmentation, tracking, recognition
- Image representation: pixels, intensity, RGB channels, tensors/matrices
- Image formation and geometry: camera model, perspective projection
- Key challenges: illumination, scale, viewpoint, occlusion, noise
- Low-level vs high-level vision

**Keywords**
- Perspective projection, camera model, illumination variation, viewpoint/scale variation, scene understanding

---

## Lecture 2 – Image Classification
**Covered topics**
- Image classification: mapping input image x to label y
- Feature representation and feature space
- Traditional pipeline: hand-crafted features → classifier
- Nearest neighbor intuition (similarity in feature space)
- Decision boundaries (conceptual)
- Training vs testing; generalization
- Curse of dimensionality

**Keywords**
- Feature space, decision boundary, generalization, curse of dimensionality

---

## Lecture 3 – Loss Functions, Optimization, Neural Networks
**Covered topics**
- Neural network basics (layers, parameters, forward pass)
- Loss functions: purpose and role
- Regression loss: MSE
- Classification loss: Cross-Entropy
- Optimization: Gradient Descent
- GD variants: Batch, Mini-batch, SGD
- Learning rate effects
- Optimizers: Momentum, Adam (conceptual)
- Training issues: local minima, saddle points, initialization sensitivity

**Keywords**
- Cross-Entropy, MSE, Gradient Descent, learning rate, mini-batch, Adam, Momentum

---

## Lecture 4 – CNN Fundamentals
**Covered topics**
- Why CNNs for images (spatial structure)
- Convolution: kernel/filter, sliding window
- Feature maps
- Stride and padding (same/valid)
- Non-linearity (ReLU) concept
- Pooling (max/average)
- Hierarchical feature learning
- Local receptive fields and weight sharing

**Keywords**
- Kernel/filter, feature map, stride, padding, pooling, weight sharing, receptive field

---

## Lecture 5 – CNN Overfitting
**Covered topics**
- Overfitting vs underfitting
- Symptoms (train vs validation behavior)
- Bias–variance tradeoff
- Model capacity and dataset size
- Mitigation methods (as concepts): data augmentation, dropout, regularization, early stopping

**Keywords**
- Bias–variance, model capacity, data augmentation, dropout, regularization, early stopping

---

## Lecture 6 – CNN Building and Tuning
**Covered topics**
- Hyperparameters (learning rate, batch size, depth, filters, kernel size)
- Epoch vs iteration
- Learning curves and diagnosis
- Training stability considerations
- Hyperparameter search: grid vs random
- Practical tuning workflow (baseline → iterate)

**Keywords**
- Hyperparameters, learning curves, epoch vs iteration, grid search, random search, training stability

---

## Lecture 7 – CNN Architectures
**Covered topics**
- Motivation for different architectures
- AlexNet (historical impact)
- VGGNet: deep stacks of 3×3 kernels
- Inception: multi-scale branches, 1×1 convolution for dimensionality reduction
- ResNet: residual learning and skip connections
- Trade-offs: depth, parameters, computation
- Vanishing gradient problem

**Keywords**
- AlexNet, VGG (3×3), Inception (multi-scale, 1×1 conv), ResNet (skip connections), vanishing gradient

---

## Lecture 8 – LSTM
**Covered topics**
- Sequence modeling and temporal dependency
- RNN limitations (vanishing gradient)
- LSTM structure: cell state, hidden state
- Gates: forget, input, output
- Why gated mechanisms work
- Application context (video/sequence data)

**Keywords**
- Temporal dependency, cell state, hidden state, forget/input/output gates, vanishing gradient

---

## Lecture 9 – Generative Models
**Covered topics**
- Autoregressive models: PixelRNN/PixelCNN
  - Pixel-by-pixel generation, raster scan order
  - Masked convolutions
- GANs:
  - Generator vs Discriminator
  - Minimax (zero-sum) game
  - Training dynamics and freezing components (conceptual)
- Common issues: instability, imbalance, mode collapse
- Evaluation: Inception Score (IS), Fréchet Inception Distance (FID)

**Keywords**
- Masked convolutions, Generator/Discriminator, minimax, mode collapse, IS, FID

---

## Lecture 10 – Encoders and Decoders (Autoencoders)
**Covered topics**
- Autoencoder pipeline: encoder → latent space → decoder
- Reconstruction loss
- Undercomplete vs overcomplete autoencoders
- Variants: denoising, stochastic, contractive (as applicable)
- Use cases: compression, denoising, representation learning

**Keywords**
- Latent space, reconstruction, denoising, representation learning

---

## Lecture 11 – Detection, Segmentation, Feature Extraction
**Covered topics**
- Object detection outputs: bounding box, class, confidence
- Detector families: two-stage vs one-stage (conceptual)
- Semantic segmentation (pixel-wise labeling)
- Instance segmentation (separate object masks)
- Classical features: HoG, SIFT
- Deep vs hand-crafted features (comparison)

**Keywords**
- Bounding box, two-stage vs one-stage, pixel-wise, instance masks, HoG, SIFT

---

## Lecture 12 – Feature Visualization and Network Inversion
**Covered topics**
- Model interpretability motivation
- Feature visualization (what layers respond to)
- Network inversion (reconstructing inputs from activations)
- DeconvNet / UpconvNet concepts
- Early vs deep layer representations
- Privacy risks
- Adversarial examples and robustness

**Keywords**
- Interpretability, network inversion, deconvnet/upconvnet, early vs deep layers, adversarial examples

---

## Lecture 13 – Learning on Videos, 3D DL, Scene Graphs
**Covered topics**
- Video understanding and temporal nature
- Motion concepts: motion detection, optical flow
- Tracking: multi-object tracking (identity over time)
- Activity recognition:
  - CNN + LSTM
  - 3D CNN (spatiotemporal convolution)
- Structured scene understanding:
  - Visual relationships (subject–predicate–object)
  - Scene graphs
  - Graph Neural Networks (GNNs)

**Keywords**
- Optical flow, MOT, activity recognition, 3D CNN, scene graph, visual relationships, GNN

---
