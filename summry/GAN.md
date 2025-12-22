#  Lecture 9: Generative Models

##  Learning Outcomes
By the end of this lecture, you should be able to:

* Explain the concept and purpose of **generative modeling** in computer vision
* Describe the **architecture and working principles** of GANs
* Differentiate clearly between **Generator** and **Discriminator** networks
* Discuss **GAN training procedures** and **evaluation metrics**
* Explore applications of generative models in **AI creativity and vision tasks**

---

##  PixelRNN & PixelCNN
PixelRNN and PixelCNN are **autoregressive generative models** that generate images **pixel by pixel** following a predefined order.

###  How They Work
* **Joint Distribution Learning**
  These models learn the joint probability of an image by factorizing it into conditional probabilities:

 $$[
  P(\mathbf{x}) = \prod_{i=1}^{n^2} P(x_i \mid x_1, \dots, x_{i-1})
  ]$$

* **Raster-Scan Order**
  Pixels are generated sequentially from **left to right** and **top to bottom**.

* **Masked Convolutions**
  Ensure that each pixel only depends on previously generated pixels:

  * **Type A mask**: used in the first layer
  * **Type B mask**: used in subsequent layers

---

### ⚖️ PixelRNN vs. PixelCNN
| Feature                 | PixelRNN                               | PixelCNN                                     |
| ----------------------- | -------------------------------------- | -------------------------------------------- |
| **Architecture**        | Recurrent Neural Networks (LSTM-based) | Convolutional Neural Networks                |
| **Generation Process**  | Fully sequential                       | Partially parallel using masked convolutions |
| **Training Speed**      | Slow                                   | Faster                                       |
| **Dependency Modeling** | Captures long-range dependencies       | Focuses on local spatial context             |

---

##  Generative Adversarial Networks (GANs)
A **Generative Adversarial Network (GAN)** consists of two neural networks trained in competition.

###  Core Components
* **Generator (G)** 

  * Takes random noise as input
  * Learns to generate images that resemble real training data

* **Discriminator (D)** 
  * Binary classifier
  * Predicts whether an input image is **real** or **fake**
  * Typically uses convolutional layers followed by a **sigmoid** output

---

##  GAN Minimax Objective
GAN training is formulated as a **minimax game** based on **zero-sum game theory**.

### Optimization Goals
* **Discriminator (D)**
  Maximizes the probability of correctly classifying real and fake images:

 $$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$
  
 * **Generator (G)**
  Minimizes the probability of the discriminator detecting fake images:

 $$
  \min_G ; \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]
  $$

This creates a **competitive equilibrium** where both models improve simultaneously.

---

## 🔄 GAN Training Process
* **Step 1: Train the Discriminator**
  * Input: real images (labeled real) + generated images (labeled fake)
  * Objective: improve classification accuracy

* **Step 2: Train the Generator**
  * Uses a **combined model (G + D)**
  * **Discriminator weights are frozen**
  * Generator updates aim to fool the discriminator

* **Epoch-wise Training**
  * Both networks are trained alternately in each epoch
  * Performance improves iteratively for both models

---

##  Upsampling in GANs
* Traditional CNNs use **pooling** to downsample images
* Generative models use **upsampling** to:

  * Increase image resolution
  * Repeat rows and columns to scale spatial dimensions

---

##  Evaluation Metrics for GANs
Since visual inspection alone is insufficient, objective metrics are used:

1. **Inception Score (IS)**

   * Evaluates:

     * Image **quality**
     * Image **diversity**
   * Higher score → better generation

2. **Fréchet Inception Distance (FID)**

   * Measures the distance between real and generated image distributions
   * **Lower FID** → generated data closer to real data

---
## **Scenario**

You are given a **Generative Adversarial Network (GAN)** trained to generate grayscale images similar to a real dataset.

The model consists of:

* A **Generator (G)** that takes random noise as input
* A **Discriminator (D)** that outputs a probability using a **sigmoid activation function**

During training, the following observations were made:

* The Discriminator quickly reaches **near-perfect accuracy**
* The Generator’s loss stops improving after a few epochs
* Generated images remain noisy and unrealistic
* The **Fréchet Inception Distance (FID)** remains high

1- Explain the **role of the Generator and the Discriminator** in the GAN architecture.

<details>
<summary><strong>Hidden Answer</strong></summary>

The Generator learns to produce fake images that resemble real data, while the Discriminator acts as a binary classifier that distinguishes between real and fake images. The Generator aims to fool the Discriminator, and the Discriminator aims to correctly classify inputs.

</details>

2- GAN training is based on a **minimax zero-sum game**.

**a)** What does *zero-sum* mean in this context?
**b)** What is the objective of the Generator and the Discriminator?

<details>
<summary><strong>Hidden Answer</strong></summary>

Zero-sum means that any improvement in one network results in a loss for the other. The Discriminator aims to maximize correct classification of real and fake images, while the Generator aims to minimize the Discriminator’s ability to detect fake images.

</details>

3- Why are the **Discriminator’s weights frozen** when training the Generator using the combined model?

<details>
<summary><strong>Hidden Answer</strong></summary>

Because the Generator and Discriminator have opposing objectives. Freezing the Discriminator ensures that only the Generator updates its weights while receiving feedback from a fixed Discriminator.

</details>

4- Based on the scenario, explain **why the Generator stops improving** when the Discriminator becomes too strong early in training.

<details>
<summary><strong>Hidden Answer</strong></summary>

When the Discriminator becomes too accurate, it easily detects fake images, resulting in very small gradients for the Generator. This prevents meaningful learning and slows or stops Generator improvement.

</details>

5- The Generator uses **upsampling layers**.

**a)** Why is upsampling required in generative models?
**b)** How does upsampling differ from pooling in CNNs?

<details>
<summary><strong>Hidden Answer</strong></summary>

Upsampling is used to increase the spatial resolution of generated images. Unlike pooling, which reduces image size and information, upsampling expands dimensions by repeating rows and columns.

</details>


6- Two evaluation metrics are used: **Inception Score (IS)** and **Fréchet Inception Distance (FID)**.

**a)** What does each metric measure?
**b)** Why does the FID remain high in this scenario?

<details>
<summary><strong>Hidden Answer</strong></summary>

IS measures image quality and diversity, while FID compares the distributions of real and generated images in a feature space. The FID remains high because generated images are still far from the real data distribution.

</details>


7- Why is **PixelCNN faster to train than PixelRNN**?

<details>
<summary><strong>Hidden Answer</strong></summary>

PixelCNN uses masked convolutions that allow partial parallelism, while PixelRNN relies on sequential recurrent processing, making PixelCNN faster.

</details>


8- If training continues without correcting the imbalance between the Generator and Discriminator:

* What is the likely outcome for the Generator?
* Name one training-related issue that may occur.

<details>
<summary><strong>Hidden Answer</strong></summary>

The Generator may fail to improve and produce low-quality outputs. One possible issue is vanishing gradients.

</details>



9- Suggest **one strategy** to help maintain balance between the Generator and Discriminator.

<details>
<summary><strong>Hidden Answer</strong></summary>

One strategy is to slow down Discriminator training so that the Generator has time to improve, maintaining a balanced minimax game.

</details>

