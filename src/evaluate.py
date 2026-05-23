"""
OmniMed Evaluation Script
=========================
Calculates ROUGE-L and Clinical BERTScore on a subset of the validation set.

Usage (on Kaggle with GPU):
    export BNB_CUDA_VERSION=121 && python src/evaluate.py \
        --num_samples 50 \
        --output_file evaluation_results.json

Dependencies (add to your Kaggle pip install cell):
    pip install rouge-score bert-score
"""

import torch
import os
import sys
import json
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

# Ensure src module is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import load_inference_model
from utils.main import clean_report


def load_test_data(num_samples=50):
    """
    Loads the validation split (same random_state=42 used during training)
    and returns a subset for evaluation.
    """
    metadata_path = "/kaggle/input/datasets/nikeshreddypatlolla/mimic-cxr-dataset/mimic-cxr-dataset/metadata.csv"
    root_dir = "/kaggle/input/datasets/nikeshreddypatlolla/mimic-cxr-dataset/mimic-cxr-dataset"

    full_df = pd.read_csv(metadata_path)
    df = full_df[full_df['ViewPosition'].isin(['PA', 'AP'])].reset_index(drop=True)

    # Use the exact same split as training so we evaluate on unseen data
    _, val_df = train_test_split(df, test_size=0.1, random_state=42)
    val_df = val_df.reset_index(drop=True)

    # Shuffle and take a subset
    eval_df = val_df.sample(n=min(num_samples, len(val_df)), random_state=123).reset_index(drop=True)

    samples = []
    skipped = 0
    for idx in range(len(eval_df)):
        row = eval_df.iloc[idx]
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = str(row['dicom_id'])
        p_prefix = f"p{subject_id[:2]}"

        img_path = os.path.join(
            root_dir, "official_data_iccv_final", "files",
            p_prefix, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg"
        )
        report_path = os.path.join(
            root_dir, "mimic-cxr-reports", "files",
            p_prefix, f"p{subject_id}", f"s{study_id}.txt"
        )

        # Skip if files are missing
        if not os.path.exists(img_path) or not os.path.exists(report_path):
            skipped += 1
            continue

        with open(report_path, 'r') as f:
            raw_text = f.read()
            ground_truth = clean_report(raw_text)

        # Skip empty reports
        if not ground_truth.strip():
            skipped += 1
            continue

        samples.append({
            "image_path": img_path,
            "ground_truth": ground_truth,
        })

    print(f"Loaded {len(samples)} valid samples for evaluation (skipped {skipped} due to missing files/empty reports)")
    return samples


def generate_predictions(model, tokenizer, device, samples):
    """
    Runs inference on each sample image and collects predictions.
    """
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])

    predictions = []
    references = []

    for i, sample in enumerate(tqdm(samples, desc="Generating reports")):
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            img_tensor = img_transforms(image).unsqueeze(0).to(device, dtype=torch.float16)

            with torch.no_grad():
                # Vision encoding (same as inference.py)
                all_features = model.vision_encoder.visual.trunk.forward_features(img_tensor)
                patch_tokens = all_features[:, 1:, :]
                projected_features = model.projector(patch_tokens)

                # Prepare text prompt
                system_prompt = "You are an expert radiologist. Analyze the chest X-ray and provide a detailed report."
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "<image>\nAnalyze this chest X-ray and provide a detailed clinical report."}
                ]
                prompt_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

                parts = prompt_text.split("<image>")
                before_img_tokens = tokenizer(parts[0], return_tensors="pt").input_ids.to(device)
                after_img_tokens = tokenizer(parts[1], return_tensors="pt").input_ids.to(device)

                before_embeds = model.llm.get_input_embeddings()(before_img_tokens)
                after_embeds = model.llm.get_input_embeddings()(after_img_tokens)

                inputs_embeds = torch.cat([before_embeds, projected_features, after_embeds], dim=1)

                outputs = model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=300,
                    temperature=0.3,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(response)
            references.append(sample["ground_truth"])

            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(samples)}] Latest prediction snippet: {response[:80]}...")

        except Exception as e:
            print(f"  [SKIP] Sample {i} failed: {e}")
            continue

    return predictions, references


def compute_rouge(predictions, references):
    """Compute ROUGE-L F1 score."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'].fmeasure)

    avg_rouge = sum(scores) / len(scores) if scores else 0.0
    return avg_rouge, scores


def compute_bertscore(predictions, references):
    """Compute BERTScore using a clinical language model."""
    from bert_score import score as bert_score

    # Use microsoft/BiomedNLP-BiomedBERT for clinical text understanding
    # Falls back to default roberta-large if not available
    try:
        P, R, F1 = bert_score(
            predictions, references,
            model_type="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            lang="en",
            verbose=True
        )
    except Exception:
        print("BiomedBERT not available, falling back to default model...")
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)

    avg_f1 = F1.mean().item()
    return avg_f1, F1.tolist()


def main():
    parser = argparse.ArgumentParser(description="Evaluate OmniMed on the validation set")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of validation samples to evaluate on (default: 100)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="Path to save the JSON results file")
    args = parser.parse_args()

    # 1. Load model
    print("=" * 60)
    print("OMNIMED EVALUATION")
    print("=" * 60)
    model, tokenizer, device = load_inference_model()

    # 2. Load test data
    print("\nLoading validation data...")
    samples = load_test_data(num_samples=args.num_samples)

    if len(samples) == 0:
        print("ERROR: No valid samples found. Check your dataset paths.")
        return

    # 3. Generate predictions
    print(f"\nRunning inference on {len(samples)} samples...")
    predictions, references = generate_predictions(model, tokenizer, device, samples)

    print(f"\nSuccessfully generated {len(predictions)} predictions.")

    # 4. Compute ROUGE-L
    print("\n--- Computing ROUGE-L ---")
    avg_rouge, rouge_scores = compute_rouge(predictions, references)
    print(f"  ROUGE-L F1: {avg_rouge:.4f}")

    # 5. Compute Clinical BERTScore
    print("\n--- Computing Clinical BERTScore ---")
    avg_bert, bert_scores = compute_bertscore(predictions, references)
    print(f"  Clinical BERTScore F1: {avg_bert:.4f}")

    # 6. Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples Evaluated : {len(predictions)}")
    print(f"  ROUGE-L F1        : {avg_rouge:.4f}")
    print(f"  Clinical BERTScore: {avg_bert:.4f}")
    print("=" * 60)

    # 7. Save detailed results to JSON
    results = {
        "num_samples": len(predictions),
        "rouge_l_f1": round(avg_rouge, 4),
        "clinical_bertscore_f1": round(avg_bert, 4),
        "per_sample": [
            {
                "prediction": pred[:200],  # Truncate for readability
                "reference": ref[:200],
                "rouge_l": round(r, 4),
                "bertscore": round(b, 4),
            }
            for pred, ref, r, b in zip(predictions, references, rouge_scores, bert_scores)
        ]
    }

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
