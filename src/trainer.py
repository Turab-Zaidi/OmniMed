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
import os
from huggingface_hub import HfApi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




def train():

    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    output_dir = "./outputs/omnimed_v1"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # BiomedCLIP specific transforms
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])

    model = OmniMedModel(model_id)
    
    model.llm.gradient_checkpointing_enable()

    full_df = pd.read_csv("/kaggle/input/datasets/nikeshreddypatlolla/mimic-cxr-dataset/mimic-cxr-dataset/metadata.csv")

    df = full_df[full_df['ViewPosition'].isin(['PA', 'AP'])].reset_index(drop=True)
    
    # 90/10 Split
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    train_dataset = MimicCxrDataset(train_df, tokenizer, img_transforms, "/kaggle/input/datasets/nikeshreddypatlolla/mimic-cxr-dataset")
    val_dataset = MimicCxrDataset(val_df, tokenizer, img_transforms, "/kaggle/input/datasets/nikeshreddypatlolla/mimic-cxr-dataset")


    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,   # 2 images per GPU = Total Batch 4
        gradient_accumulation_steps=4,    # Effective Batch Size = 16
        warmup_steps=50,
        max_steps=1000,                   # Adjust based on your time limit
        learning_rate=2e-4,               # LoRA usually needs higher LR
        fp16=True,                        # Use Mixed Precision for T4
        logging_steps=10,

        evaluation_strategy="steps",        # Eval every X steps
        eval_steps=100,                     # Run validation every 100 steps
        save_strategy="steps",              # Save every X steps
        save_steps=100,
        save_total_limit=2,                 # ONLY KEEP 2 BEST CHECKPOINTS (Saves Disk)
        load_best_model_at_end=True,        # Load the best version after training
        metric_for_best_model="loss",       # Best model = lowest val loss
        greater_is_better=False,

        report_to="wandb",                # Track progress on your phone/PC
        remove_unused_columns=False,      # Important: Keep 'images' column
        ddp_find_unused_parameters=False  # Required for DDP + LoRA
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset, 

    )

    # 7. Start Training
    print("Starting training on 2x T4 GPUs...")
    trainer.train()

    if trainer.is_world_process_zero():
        model.llm.save_pretrained(f"{output_dir}/lora_adapters")
        torch.save(model.projector.state_dict(), f"{output_dir}/projector.pt")
        print(f"Training complete. Weights saved to {output_dir}")
    
    return model, tokenizer


def save_and_push(model, tokenizer, repo_id):
    """
    Saves the small LoRA adapters and the Projector layer to Hugging Face.
    """
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
    print(f" Model successfully pushed to: https://huggingface.co/{repo_id}")



if __name__ == "__main__":
    model, tokenizer = train() 
    
    repo_id = "Turab0104/OmniMed-CXR-Llama3" 
    if os.environ.get("LOCAL_RANK", "0") == "0":
        repo_id = "Turab0104/OmniMed-CXR-Llama3" 
        save_and_push(model, tokenizer, repo_id)
    