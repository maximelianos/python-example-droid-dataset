Classifier-Free Guidance (CFG) is a sampling technique used in **diffusion models** (especially **text-to-image models** like Stable Diffusion) to **steer generation toward desired outputs (e.g., a text prompt)** without needing an external classifier.

Here’s a technical breakdown suitable for someone with a strong computer vision background:

---

### 🔧 Core Concept

Diffusion models sample from a distribution $p_\theta(x_t | x_{t+1}, c)$, where:

* $x_t$: a noisy sample at timestep $t$,
* $c$: conditioning information (e.g., a class label or text embedding),
* $\theta$: model parameters.

In **Classifier-Free Guidance**, we train a **single model** to predict denoised samples with and without conditioning:

* **Conditioned**: $\epsilon_\theta(x_t, c)$
* **Unconditioned**: $\epsilon_\theta(x_t, \emptyset)$

During sampling, we **interpolate** between these two predictions to guide the sample toward the desired condition:

$$
\epsilon_{\text{guided}} = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))
$$

Where:

* $w$ is the **guidance scale** (usually >1),
* Increasing $w$ sharpens fidelity to the condition (e.g., makes images more aligned to the text) at the cost of diversity.

---

### 🧠 Intuition

It mimics what external classifier guidance does (e.g., in DDPMs using gradients from a classifier), **but avoids the need for a separate classifier**. Instead, you:

* Train the model on both conditioned and unconditioned objectives by randomly dropping the condition during training (often with 10–20% probability).
* At inference, blend predictions from both conditioned and unconditioned paths to guide sampling.

---

### 🔬 Comparison to Classifier Guidance

| Aspect              | Classifier Guidance        | Classifier-Free Guidance                |
| ------------------- | -------------------------- | --------------------------------------- |
| Requires classifier | ✅ Yes                      | ❌ No                                    |
| Simplicity          | ❌ Complex setup            | ✅ Simpler training/inference            |
| Flexibility         | 🔴 Limited to known labels | 🟢 Flexible (e.g., text, embeddings)    |
| Performance         | Good, but brittle          | Often state-of-the-art in text-to-image |

---

### 📈 Use in Modern Models

CFG is central to:

* **Stable Diffusion**
* **GLIDE**
* **Imagen**
* **DALL·E 2**

It allows precise control over conditioning (like text prompts), which is critical in generative image models.

---

Let me know if you'd like a visual or code-level explanation of CFG in PyTorch!
