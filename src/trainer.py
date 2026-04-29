import os
import pandas as pd
import sys
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
)
from src.model import OmniMedModel
from src.dataset import MimicCxrDataset
import open_clip
from torchvision import transforms
from huggingface_hub import HfApi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



class OmniMedTrainer(Trainer):


    def _save(self, output_dir, state_dict=None):
        os.makedirs(output_dir, exist_ok=True)
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # 1. Save ONLY LoRA adapters (~50 MB)
        model.llm.save_pretrained(os.path.join(output_dir, "lora_adapters"))

        # 2. Save ONLY projector weights (~40 MB)
        torch.save(model.projector.state_dict(), os.path.join(output_dir, "projector.pt"))

    def _load_from_checkpoint(self, checkpoint_path):
        """Override loading to match our custom save format."""
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Load LoRA adapters
        from peft import PeftModel
        # The base LLM is already loaded in __init__, we just load the adapters
        adapter_path = os.path.join(checkpoint_path, "lora_adapters")
        if os.path.exists(adapter_path):
            model.llm = PeftModel.from_pretrained(model.llm.base_model, adapter_path)

        # Load projector weights
        projector_path = os.path.join(checkpoint_path, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))



def train():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "./outputs/omnimed_v1"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])

    model = OmniMedModel(model_id)
    model.llm.gradient_checkpointing_enable()

    full_df = pd.read_csv("/kaggle/input/datasets/nikeshreddypatlolla/mimic-cxr-dataset/mimic-cxr-dataset/metadata.csv")
    df = full_df[full_df['ViewPosition'].isin(['PA', 'AP'])].reset_index(drop=True)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = MimicCxrDataset(train_df, tokenizer, img_transforms, "/kaggle/input/datasets/nikeshreddypatlolla/mimic-cxr-dataset/mimic-cxr-dataset")
    val_dataset = MimicCxrDataset(val_df, tokenizer, img_transforms, "/kaggle/input/datasets/nikeshreddypatlolla/mimic-cxr-dataset/mimic-cxr-dataset")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        max_steps=1000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,

        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        per_device_eval_batch_size=2,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,

        report_to="wandb",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # USE THE CUSTOM TRAINER HERE
    trainer = OmniMedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting training on 2x T4 GPUs...")
    trainer.train()

    if trainer.is_world_process_zero():
        model.llm.save_pretrained(f"{output_dir}/lora_adapters")
        torch.save(model.projector.state_dict(), f"{output_dir}/projector.pt")
        print(f"Training complete. Weights saved to {output_dir}")

    return model, tokenizer


def save_and_push(model, tokenizer, repo_id):
    local_dir = "./final_model"
    os.makedirs(local_dir, exist_ok=True)

    model.llm.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    torch.save(model.projector.state_dict(), f"{local_dir}/projector.pt")

    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message="End of Kaggle Training: SOTA MedVLM weights"
    )
    print(f"Model successfully pushed to: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    model, tokenizer = train()

    repo_id = "Turab0104/OmniMed-CXR-Llama3"
    if os.environ.get("LOCAL_RANK", "0") == "0":
        save_and_push(model, tokenizer, repo_id)
    