import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import open_clip

class OmniMedModel(nn.Module):
    def __init__(self, model_id="meta-llama/Llama-3.1-8B-Instruct"):
        super().__init__()

        self.vision_encoder, _, _ = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.vision_encoder.visual.output_tokens = True 

        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096)
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto", # Automatically handles 2x T4 balance
            trust_remote_code=True
        )

        self.llm = prepare_model_for_kbit_training(self.llm)
        lora_config = LoraConfig(
            r=16, 
            lora_alpha=32, 
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)


    def forward(self, images, input_ids, attention_mask, labels=None):

        with torch.no_grad():
            # Get the 256 patch tokens [Batch, 256, 512]
            _, patch_tokens = self.vision_encoder.visual(images)
        
        image_tokens = self.projector(patch_tokens) 

        text_embeds = self.llm.get_input_embeddings()(input_ids)

        combined_embeds = torch.cat((image_tokens, text_embeds), dim=1)

        batch_size = images.shape[0]
        visual_mask = torch.ones((batch_size, 256), device=images.device)
        full_attention_mask = torch.cat((visual_mask, attention_mask), dim=1)

        if labels is not None:
            visual_labels = torch.full((batch_size, 256), -100, device=labels.device)
            full_labels = torch.cat((visual_labels, labels), dim=1)
        else:
            full_labels = None

        outputs = self.llm(
            inputs_embeds=combined_embeds, 
            attention_mask=full_attention_mask,
            labels=full_labels
        )
        
        return outputs