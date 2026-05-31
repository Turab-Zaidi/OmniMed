# 🩺 OmniMed — Multimodal Medical Vision-Language Model

> A custom-built Vision-Language Model (VLM) that analyzes chest X-rays and generates professional radiology reports using the LLaVA architecture pattern — trained from scratch on dual T4 GPUs via QLoRA.

---

## 📌 Overview

**OmniMed** is an end-to-end multimodal AI system that fuses a specialized medical vision encoder (BiomedCLIP) with a large language model (Llama-3 8B Instruct) to generate clinically descriptive reports from chest X-ray images.

The model was trained on the **MIMIC-CXR** dataset using parameter-efficient fine-tuning (QLoRA + LoRA) to make the entire pipeline runnable on consumer-grade hardware, while producing medically coherent, structured radiology reports.

| Property | Detail |
|---|---|
| **Task** | Chest X-Ray → Radiology Report Generation |
| **Dataset** | MIMIC-CXR (chest radiographs + free-text reports) |
| **Vision Encoder** | BiomedCLIP ViT-B/16 (frozen) |
| **Language Model** | Llama-3.1 8B Instruct (4-bit QLoRA) |
| **Training Hardware** | Dual NVIDIA T4 GPUs (Kaggle) |
| **Trained Weights** | [Turab0104/OmniMed-CXR-Llama3](https://huggingface.co/Turab0104/OmniMed-CXR-Llama3) |

---

## 🏆 Evaluation Results

Evaluated on **100 held-out validation samples** from MIMIC-CXR (never seen during training):

| Metric | Score | Interpretation |
|---|---|---|
| **Clinical BERTScore F1** | 0.835 | Strong semantic alignment — the model understands medical concepts, not just words |

> **Why Clinical BERTScore?** Standard word-overlap metrics like ROUGE penalize medically correct paraphrases. If the model writes "enlarged cardiac silhouette" but the ground truth says "cardiomegaly," ROUGE gives a score of 0 — even though both phrases mean the same thing clinically. Clinical BERTScore uses **BiomedBERT** to compare the *semantic meaning* of reports rather than their exact wording, making it a far more appropriate metric for medical text generation.

---

## 🏗️ Architecture

OmniMed follows the **LLaVA architectural pattern** — the same approach used in GPT-4V, LLaVA, and Gemini for multimodal reasoning.

```
[Chest X-Ray (224x224)]
        │
        ▼
┌─────────────────────┐
│  BiomedCLIP ViT-B   │  ← Frozen. Pre-trained on 15M medical image-text pairs.
│  (Vision Encoder)   │    Extracts 256 patch-level feature vectors.
└─────────────────────┘
        │
        │  256 × 768 patch embeddings
        ▼
┌─────────────────────┐
│   MLP Projector     │  ← Fully trainable. The key new component.
│  (Linear→GELU→Lin) │    Translates visual features into LLM token space.
└─────────────────────┘
        │
        │  256 × 4096 projected visual tokens
        ▼
┌──────────────────────────────────────────┐
│   [IMG_1 ... IMG_256 | TEXT_1 ... TEXT_N] │
│                                          │
│        Llama-3.1 8B Instruct             │  ← 4-bit quantized (QLoRA).
│      (Language Model + LoRA)             │    Only LoRA adapters are trained.
└──────────────────────────────────────────┘
        │
        ▼
 Generated Radiology Report
```

### Why This Design?

- **Frozen Vision Encoder:** BiomedCLIP was pre-trained by Microsoft on 15 million biomedical image-caption pairs (PMC-15M). Freezing it preserves its expert medical knowledge while dramatically reducing VRAM usage.
- **Trainable MLP Projector:** This is the bridge between the visual world and the language world. It learns to "translate" medical image patches into concepts Llama-3 can reason about.
- **QLoRA on Llama-3:** Loading the 8B model in 4-bit quantization reduces VRAM from ~32GB to ~5GB. LoRA adapters (r=16, alpha=32) are injected into all 4 attention projections (q, k, v, o), adding <1% additional trainable parameters while significantly adapting the LLM's behavior.

---

## 🔬 Qualitative Analysis

### Strengths
- **Medical Fluency:** Every generated report reads like it was written by a radiologist — correct clinical terminology, proper Findings/Impression structure, appropriate hedging language.
- **Normal X-Ray Detection:** Excellent performance on healthy chests (best BERTScore: 0.887). The model reliably rules out acute pathology.
- **Cardiomegaly Detection:** Consistently identifies enlarged cardiac silhouettes and correctly contextualizes post-surgical patients (sternotomy, CABG).
- **Temporal Language:** Naturally produces radiologist-style comparison phrases like "in comparison with the study of ___", "interval change", "unchanged."

### Known Limitations
1. **Normal Default Bias (~25-30% of cases):** The model sometimes defaults to a "lungs are clear / no acute process" report even when the X-ray contains pathology. This is a direct consequence of class imbalance in MIMIC-CXR (majority of real-world chest X-rays are normal).

2. **Device/Line Detection is Inconsistent:** The model occasionally hallucinates surgical hardware (e.g., CABG sternotomy wires) not present in the image, and can miss real devices like PICC lines.

3. **Severity Precision:** When pathology is detected, severity grading can be imprecise (e.g., "moderate" vs "mild-to-moderate" cardiomegaly).

### Root Cause Analysis
The normal bias stems from three compounding factors:
- **Dataset Imbalance:** ~70-80% of MIMIC-CXR images are normal/near-normal; the model learned to minimize loss by "playing the statistical odds."
- **Frozen Vision Encoder + Resolution Bottleneck:** Resizing X-rays to 224×224 blurs subtle findings (small pneumothorax, hairline fractures). Without unfreezing the ViT, these nuances can't be learned.
- **Cross-Entropy Loss Does Not Prioritize Disease Words:** The loss function penalizes missing the word "pneumothorax" and missing the word "the" equally. Critical clinical terms are numerically overwhelmed by fluency loss.

---

## 📁 Project Structure

```
OmniMed/
├── src/
│   ├── model.py          # OmniMedModel — architecture (ViT + Projector + LLM)
│   ├── dataset.py        # MimicCxrDataset — data loading, label masking
│   ├── trainer.py        # OmniMedTrainer — custom save/load for decoupled weights
│   ├── inference.py      # Full inference pipeline
│   └── evaluate.py       # Clinical BERTScore evaluation
├── utils/
│   └── main.py           # clean_report() — regex extraction of Findings/Impression
├── assets/
│   └── sample_xray.png   # Sample X-ray for demo
├── app.py                # Gradio web interface
├── requirements.txt
└── README.md
```

---

## ⚙️ Key Implementation Details

### Label Masking Strategy
A critical design decision for correct VLM training: loss is **only computed on the assistant's answer tokens**. The image tokens (256 positions) and user prompt tokens are masked to `-100` (PyTorch's `ignore_index`). This prevents the model from wasting capacity learning to "predict" the image or the input question.

```
Input sequence:  [IMG×256 | User Prompt | Assistant Answer | <eos>]
Labels:          [-100×256 | -100×N     | Actual tokens    | eos_id]
                  ^^^^^^^^   ^^^^^^^^^   ^^^^^^^^^^^^^^^^
                  Masked      Masked      Loss computed here
```

### Decoupled Weight Saving
Standard HuggingFace `Trainer.save_model()` doesn't handle our custom architecture (a PEFT-wrapped LLM + a separate projector). The custom `OmniMedTrainer._save()` method saves them separately:
- `adapter_model.safetensors` — The LoRA delta weights only (~300MB total instead of 16GB)
- `projector.pt` — The MLP bridge weights

This allows the full model to be reconstructed at inference time with no full-weight copies stored.

### DDP-Compatible Model Loading
The `OmniMedModel` detects `LOCAL_RANK` at initialization to decide between `device_map="auto"` (single-GPU / model-parallel inference) and no explicit device map (letting `accelerate` handle DDP wrapping for multi-GPU training).

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install bitsandbytes>=0.43.1
```

### 2. Run Inference on a Chest X-Ray

> ⚠️ Requires access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face. Request access [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) first.

```bash
# Set your HF token
export HF_TOKEN=your_token_here

# On Linux/Mac
export BNB_CUDA_VERSION=121 && python src/inference.py --image assets/sample_xray.png

# On Windows
set BNB_CUDA_VERSION=121 && python src/inference.py --image assets/sample_xray.png
```

### 3. Run the Gradio Web Interface
```bash
python app.py
```

### 4. Run Evaluation
```bash
# Install evaluation dependencies
pip install rouge-score bert-score

# Run on 100 validation samples
export BNB_CUDA_VERSION=121 && python src/evaluate.py --num_samples 100 --output_file results.json
```

---

## 📊 Training Details

| Hyperparameter | Value |
|---|---|
| Base LLM | meta-llama/Llama-3.1-8B-Instruct |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Target Modules | q_proj, k_proj, v_proj, o_proj |
| Learning Rate | 2e-4 |
| Batch Size (effective) | 8 (2 per device × 4 gradient accumulation steps) |
| Total Training Steps | ~3,500 (1 full epoch) |
| Hardware | 2× NVIDIA T4 16GB (Kaggle) |
| Training Time | ~5 hours across 3 Kaggle sessions |
| Final Training Loss | ~1.05 |


