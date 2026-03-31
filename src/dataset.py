import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os
from utils.main import clean_report

class MimicCxrDataset(Dataset):
    def __init__(self, csv_file, tokenizer, transforms, root_dir):
        # Load metadata and filter for Frontal views (PA or AP)
        full_df = pd.read_csv(csv_file)
        self.df = full_df[full_df['ViewPosition'].isin(['PA', 'AP'])].reset_index(drop=True)
        
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.root_dir = root_dir 

    def __len__(self):
        return len(self.df)

    def get_path(self, row, file_type="image"):
        """Constructs the nested path used by MIMIC-CXR"""
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = str(row['dicom_id'])
        p_prefix = f"p{subject_id[:2]}" 
        
        if file_type == "image":
            return os.path.join(
                self.root_dir, "official_data_iccv_final", "files",
                p_prefix, f"p{subject_id}", f"s{study_id}", f"{dicom_id}.jpg"
            )
        else: # report
            return os.path.join(
                self.root_dir, "mimic-cxr-reports", "files",
                p_prefix, f"p{subject_id}", f"s{study_id}.txt"
            )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        img_path = self.get_path(row, "image")
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        # 2. Load Report Text from .txt file
        report_path = self.get_path(row, "report")
        with open(report_path, 'r') as f:
            raw_text = f.read()
            report_text = clean_report(raw_text)


        
        user_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" \
                      f"Analyze this chest X-ray and provide a detailed clinical report.<|eot_id|>" \
                      f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        full_text = user_prompt + report_text + "<|eot_id|>"

        prompt_tokens = self.tokenizer(user_prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens['input_ids'])

        full_tokens = self.tokenizer(
            full_text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=256 
        )

        # 4. Create the Labels
        # Start by copying the input_ids
        labels = full_tokens["input_ids"].squeeze().clone()
        
        # Mask the prompt part!
        labels[:prompt_len] = -100 
        
        # Mask the padding tokens as well
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "images": image,
            "input_ids": full_tokens["input_ids"].squeeze(),
            "attention_mask": full_tokens["attention_mask"].squeeze(),
            "labels": labels 
        }

