# Lecture 8 Summary: LSTM for Computer Vision

In all previous lectures, we focused on **CNNs**, which are masters of understanding *spatial* data (a single, static image).

This lecture introduces a new challenge: **sequential data**, where *order and time matter*. This includes:
* **Video:** A sequence of images (frames).
* **Language:** A sequence of words (e.g., an image caption).

CNNs have no "memory" and cannot understand this. This is why we need a new type of network: **Recurrent Neural Networks (RNNs)**.

---

## 1. Recurrent Neural Networks (RNNs)

* **What they are:** A type of neural network designed for **sequences**.
* **Core Idea:** An RNN has a **"loop"** 🔄. When it processes an input (like a word or a video frame), it also receives the **"hidden state"** (the "memory") from the *previous* step.
* **How it works:** It combines the *current input* with the *previous memory* to create a *new memory*.
* **Formula:** `new_memory = f(previous_memory, current_input)`



---

## 2. The Problem with RNNs: Short-Term Memory

* Standard RNNs suffer from the **Vanishing Gradient Problem** (but over *time*).
* **The Problem:** The "memory" from early steps (e.g., the first word in a long sentence) fades away by the time the network reaches the later steps.
* **Result:** The network has a very **short-term memory**. It can't connect the word "cat" to a pronoun "it" that appeared 20 words earlier.

---

## 3. The Solution: LSTM (Long Short-Term Memory)

An LSTM is a "smart" and powerful type of RNN that solves the vanishing gradient problem. It's designed to have **both long and short-term memory**.

* **Core Idea:** An LSTM has an internal **"Cell State"** (`C_t`). Think of this as a "conveyor belt" or a "memory highway" where information can flow for a very long time without degrading.
* **Key Innovation: "Gates"**
    * An LSTM has special "gates" (which are just small neural networks) that *learn* to control this memory.
    1.  **Forget Gate 🧠:** Decides what *old* information to **throw away** from the cell state.
    2.  **Input Gate 📥:** Decides what *new* information from the current input to **store** in the cell state.
    3.  **Output Gate 📤:** Decides what part of the cell state to **output** as the new "hidden state" (the short-term memory).



This gate system allows the LSTM to "remember" important information (like a "subject" in a sentence) for hundreds of steps, while "forgetting" irrelevant details.

---

## 4. Applications in Computer Vision (CNN + LSTM)

The real power comes when we combine **CNNs (for vision)** with **LSTMs (for sequences)**.

### Application 1: Video Analysis (Action Recognition)
* **Goal:** Classify an action in a video (e.g., "playing guitar", "running").
* **How it works (CNN + LSTM):**
    1.  **CNN:** We use a pre-trained CNN (like VGG or ResNet) to analyze **each frame** of the video. The CNN outputs a *feature vector* for each frame.
    2.  **LSTM:** We feed this **sequence of feature vectors** (one per frame) into an LSTM.
    3.  **Result:** The LSTM reads the "story" told by the features over time and makes a final classification of the action.



### Application 2: Image Captioning (The Encoder-Decoder Model)
* **Goal:** Generate a natural language sentence describing an image.
* **How it works (Encoder-Decoder):**
    1.  **Encoder (CNN):** A CNN acts as the "encoder." It looks at the image and "encodes" all its visual features into a single **context vector** (a summary).
    2.  **Decoder (LSTM):** An LSTM acts as the "decoder." It takes the CNN's context vector as its *initial memory state*. It then generates the caption, **one word at a time**, with each new word being influenced by the previous word and the image summary.



---

## 5. Advanced Topic: The Attention Mechanism

* **The Problem:** In the basic Encoder-Decoder model, the *entire* image is compressed into *one single vector*. This is a huge "bottleneck" and forces the LSTM to "remember" everything about the image at all times.
* **The Solution (Attention):**
    * We allow the Decoder (LSTM) to be "smarter."
    * At **each step** of generating a word, the LSTM is allowed to "look back" and **"pay attention"** to *different parts* of the original image's feature map.
    * **Example:** When generating the word "dog," the model "attends" to the region of the image containing the dog. When generating "frisbee," it shifts its "attention" to the frisbee.
    * This is much more powerful and is the basis for modern "Vision and Language" models (like Visual Question Answering - VQA).

