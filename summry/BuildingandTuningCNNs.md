# Lecture 6 Summary: Building and Tuning CNNs

After building the CNN **Architecture** [Lec 4](cnn2.md) and learning how to fight **Overfitting** [Lec 5](CNN_Overfitting_Regularization.md), this lecture provides a practical, step-by-step **Roadmap** 🛠️ for building a successful deep learning project from start to finish.

**The Project Roadmap:**
`Metrics -> Baseline -> Evaluation -> Tuning -> Regularization`

---

## 1. Performance Metrics (How to Measure Success)

The first step is defining "what does success look like?"

* **The Problem with "Accuracy":** Accuracy alone can be very **misleading**, especially with **Unbalanced Data**.
* **Example (Unbalanced Data):** Imagine a rare disease dataset where 99% of images are "Healthy" and 1% are "Sick".
    * A "lazy" model that always predicts "Healthy" will have **99% accuracy**.
    * This *looks* great, but the model is completely *useless* because it failed to find a single "Sick" case (0% success on the rare class).
* **The Solution:** We must use better metrics:
    * **Precision:** Of all the times the model *predicted* "Sick", how many times was it correct? (Measures "confidence").
    * **Recall:** Of all the *actual* "Sick" cases, how many did the model successfully *find*? (Measures "completeness").



---

## 2. The Baseline Model

* **The Rule:** **"Start with simple architecture."**
* **The Goal:** Avoid "early complexity." Don't build a massive, 100-layer network from day one.
* Build a simple "baseline" model (e.g., a few CONV/POOL layers). This gives you a "reference performance" (e.g., 60% accuracy). Your job is now to "tune" your model to *beat* this 60% baseline.

---

## 3. Evaluation & Data Preparation

Before training, we must prepare our data and our evaluation strategy.

### A. Data Splitting
We split our data into 3 distinct sets:
* **Training Set (e.g., 70%):** The largest set, used for the model to "learn" the patterns.
* **Validation Set (e.g., 15%):** Used to "tune" the model. We check performance on this set to make decisions (e.g., "Should I stop training now?").
* **Test Set (e.g., 15%):** Used *only once* at the very end to give a final, unbiased score of our *best* model on data it has never seen.

### B. Preprocessing
The most critical step is **Normalization**.
* **What it is:** Scaling all input features (like pixel values) to be in a consistent range (e.g., between 0 and 1, or with a mean of 0).
* **Why it's Crucial:** As the graph shows, non-normalized data makes the "loss landscape" very "stretched." This forces **Gradient Descent** to take a slow, zig-zag path 🐢. Normalized data creates a "rounder" landscape, allowing Gradient Descent to find the solution **much faster and more directly** ⚡️.



### C. Monitoring Learning Curves
We plot the Training Loss and Validation Loss vs. Epochs.
* **Graph A (Overfitting):** This is our "diagnosis." The Training loss (blue) keeps dropping, but the Validation loss (red) **starts to increase**. This is the signal that our model is "memorizing" and we need to "tune" it.
* **Graph C (Good Fit):** The ideal. Both training and validation loss decrease and converge together.



---

## 4. Tuning (Hyperparameters & The Gradient Descent Engine)

Once we diagnose a problem (like Overfitting in Graph A), we "tune" the "hyperparameters" to improve the core **Gradient Descent** engine.

### A. Learning Rate (LR)
* This is the **most important hyperparameter**. It controls the "step size" of Gradient Descent.
* **Too High:** The optimizer "bounces" around and overshoots the minimum. The loss will never converge.
* **Too Low:** The training is **very slow** 🐢 and may get "stuck" in a bad local minimum before reaching the true solution.



### B. Optimizers
* **SGD (Stochastic Gradient Descent):** The "basic" optimizer with a fixed learning rate.
* **Adam (Most Popular):** A "smart" (adaptive) optimizer. It *automatically adjusts* the learning rate during training (starting larger for speed, getting smaller for precision).

### C. Batch Normalization (Batch Norm)
* **What it is:** A "layer" (like `CONV` or `FC`) that we add *inside* the network (e.g., after `CONV`, before `ReLU`).
* **What it does:** It **re-normalizes** the data *between* layers.
* **Why it's essential:**
    1.  **Speeds up training dramatically.**
    2.  **Stabilizes training,** allowing us to use higher Learning Rates safely.
    3.  **Acts as a Regularizer,** sometimes reducing the need for Dropout.

### D. Batch Size
* **What it is:** The number of images the model sees before making one "update" to its weights.
* **The Trade-off:**
    * **Small Batch (e.g., 32):** Updates are "noisy" (based on few samples). This "noise" can act as a regularizer and help the model find a *better* solution that **generalizes well**.
    * **Large Batch (e.g., 512):** Updates are "stable" and computationally *faster* (good for GPUs). However, they can sometimes get "stuck" in *worse* solutions (sharp minima) that don't generalize as well.

---

## 5. Regularization

After *tuning* all the above, if the model is *still* overfitting, we apply the strong **Regularization** techniques from [Lecture 5](CNN_Overfitting_Regularization.md) (e.g., **Dropout** and **L2 Regularization**) to solve it.
