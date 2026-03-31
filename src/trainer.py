import os
import torch
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from model import OmniMedModel
from dataset import MimicCxrDataset
import open_clip
from torchvision import transforms

def train():

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
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

    train_dataset = MimicCxrDataset(
        csv_file="/kaggle/input/mimic-cxr-subset/train.csv",
        tokenizer=tokenizer,
        transforms=img_transforms,
        img_dir="/kaggle/input/mimic-cxr-subset/images"
    )

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
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="no",
        report_to="wandb",                # Track progress on your phone/PC
        remove_unused_columns=False,      # Important: Keep 'images' column
        ddp_find_unused_parameters=False  # Required for DDP + LoRA
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 7. Start Training
    print("Starting training on 2x T4 GPUs...")
    trainer.train()

    model.llm.save_pretrained(f"{output_dir}/lora_adapters")
    torch.save(model.projector.state_dict(), f"{output_dir}/projector.pt")
    print(f"Training complete. Weights saved to {output_dir}")

if __name__ == "__main__":
    train()