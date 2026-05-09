

---

# OmniMed: A Multimodal Medical Vision-Language Model

## 1. Project Goal

The primary objective of this project is to build **OmniMed**, a sophisticated Vision-Language Model (VLM) capable of understanding medical images (specifically chest X-rays) and answering natural language questions about them. The architecture is designed to be efficient for fine-tuning on consumer-grade hardware (like Kaggle's dual T4 GPUs) by leveraging state-of-the-art techniques like QLoRA.

The final model will take a medical image and a text prompt as input and generate a relevant, text-based response.

## 2. Core Architecture

The OmniMed model is a custom-built VLM that fuses a specialized vision encoder with a powerful large language model.

### 2.1. The "Eyes": BiomedCLIP's Vision Transformer (ViT)

-   **Model:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
-   **Component Used:** We are **only** using the Vision Transformer (ViT) component of BiomedCLIP, accessed via `vision_encoder.visual`. The `PubMedBERT` text encoder is loaded but not used in the final model architecture.
-   **Function:** The ViT acts as a specialized feature extractor. It has been pre-trained by Microsoft on 15 million medical image-caption pairs (the PMC-15M dataset), making it an expert at converting medical image pixels into meaningful mathematical representations (embeddings).
-   **State:** The ViT is **completely frozen** (`requires_grad=False`) during our fine-tuning process. This saves a massive amount of VRAM and computation, treating the ViT as a fixed, expert "eye."

### 2.2. The "Brain": Llama-3 8B Instruct

-   **Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
-   **Function:** This serves as the reasoning and text-generation engine of our model. It takes the visual information and the user's text prompt and formulates a coherent answer.
-   **Efficiency Techniques:**
    -   **QLoRA (4-bit Quantization):** The massive 8-billion-parameter Llama-3 model is loaded in a 4-bit quantized format using `BitsAndBytesConfig`. This dramatically reduces its memory footprint from ~32GB to ~5GB, making it feasible to run on a 16GB T4 GPU.
    -   **LoRA (Low-Rank Adaptation):** Instead of fine-tuning the entire 8B model, we freeze it and only train small, low-rank "adapter" matrices injected into its self-attention layers. This reduces the number of trainable parameters by over 99%, making training fast and memory-efficient.

### 2.3. The "Bridge": A Trainable Projection Layer

-   **Architecture:** A simple Multi-Layer Perceptron (MLP): `Linear -> GELU -> Linear`.
-   **Function:** This is the most critical *new* component we are training. It acts as a translator or "bridge" between the vision and language domains.
    -   **Input:** The patch embeddings from the BiomedCLIP ViT.
    -   **Output:** Embeddings in a dimension that Llama-3 can understand (4096).
-   **State:** This component is fully **trainable**. Its primary job is to learn how to map the visual concepts from the "eyes" into the "language of thought" used by the "brain."

## 3. The Data Flow & Architectural Pattern (The LLaVA Approach)

We adopted the state-of-the-art "Visual Tokens" architectural pattern, popularized by models like LLaVA and used in GPT-4V. This approach avoids the complexity of older cross-attention methods (like Flamingo).

**The `forward` pass works as follows:**

1.  **Extract Patch Tokens:** The ViT processes a `224x224` image and outputs **256 patch tokens**. Each token represents a `16x16` region of the image. This preserves the spatial information of the image, preventing the "information bottleneck" of using a single summary vector.
2.  **Project Visual Tokens:** Our trainable `projector` takes these 256 patch tokens and translates each one into the 4096-dimensional space of Llama-3.
3.  **Get Text Embeddings:** The user's text prompt (`input_ids`) is converted into 4096-dimensional text embeddings by Llama-3's embedding layer.
4.  **Concatenate:** The 256 projected image tokens are **prepended** to the text embeddings, creating a single, long sequence. The final input to the LLM looks like: `[IMG_1, ..., IMG_256, TEXT_1, TEXT_2, ...]`.
5.  **LLM Processing:** Llama-3 processes this combined sequence. Its native self-attention mechanism allows it to "look back" at the image tokens at every step of text generation, treating the image as part of its context window.

## 4. Key Training & Implementation Details

### 4.1. The Dataset and Preprocessing

-   **Dataset:** MIMIC-CXR, a large dataset of chest X-rays and their corresponding free-text radiology reports.
-   **Text Cleaning:** A regex function `clean_report` is used to reliably extract only the "Findings" and "Impression" sections from the raw reports, as these contain the most valuable diagnostic information.
    -   The regex pattern `findings:(.*?)(impression:|...)` uses a non-greedy wildcard `.*?` with the `re.DOTALL` flag to capture multiline text within specific sections.

### 4.2. The Loss Function (Crucial Label Masking)

To ensure the model learns the correct task (answering questions about an image) and not the wrong ones (predicting the image or the user's prompt), we use a specific label masking strategy.
-   **The Rule:** Loss is only calculated for the **assistant's answer tokens**.
-   **The Mechanism:** We use PyTorch's `ignore_index` of `-100` for all other tokens in the sequence.
-   **Implementation:** The `labels` tensor passed to the LLM is structured as follows:
    -   **Image Tokens:** The first 256 positions are filled with `-100`.
    -   **User Prompt Tokens:** The positions corresponding to the user's question are also filled with `-100` (this logic is handled in the `MimicCxrDataset` class during data preprocessing).
    -   **Assistant Answer Tokens:** These positions contain the actual token IDs for the ground-truth answer.

### 4.3. The Training Script (`train.py`)

-   **Framework:** Hugging Face `Trainer`.
-   **Hardware Strategy:** The configuration is optimized for a dual T4 GPU setup on Kaggle.
    -   `device_map="auto"` handles **Model Parallelism**, splitting the large Llama-3 model across both GPUs.
    -   `per_device_train_batch_size` and `gradient_accumulation_steps` are used to achieve a larger effective batch size without causing Out of Memory (OOM) errors.
-   **LoRA Configuration:** A standard and robust configuration was chosen:
    -   `r=16`, `lora_alpha=32`: A common "sweet spot" ratio for effective learning.
    -   `target_modules`: Targets all four attention projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) for maximum impact on the model's attention mechanism.
-   **Saving Artifacts:** The final script saves only the trainable components: the LoRA adapter weights and the projector's state dictionary. This results in a small, portable model artifact (~300MB) instead of the full 8B model.
