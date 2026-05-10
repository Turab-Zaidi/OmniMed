import torch
import os
import sys

# Add the parent directory to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from src.model import OmniMedModel
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download, login

# Optional: Log in to Hugging Face if your repo is private
# Make sure your HF_TOKEN is set in your environment variables, or run `huggingface-cli login`
# login(token=os.environ.get("HF_TOKEN"))

def load_inference_model(repo_id="Turab0104/OmniMed-CXR-Llama3"):
    print("Loading base Llama-3 model...")
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize our custom architecture
    model = OmniMedModel(model_id)
    
    print(f"Downloading trained adapters and projector from {repo_id}...")
    
    # Load LoRA Adapters
    adapter_path = hf_hub_download(repo_id=repo_id, filename="adapter_model.safetensors")
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    adapter_state_dict = load_file(adapter_path)
    set_peft_model_state_dict(model.llm, adapter_state_dict)
    
    # Download and load Projector weights
    projector_path = hf_hub_download(repo_id=repo_id, filename="projector.pt")
    model.projector.load_state_dict(torch.load(projector_path, map_location="cpu"))
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def generate_report(model, tokenizer, device, image_path, prompt="Describe the findings in this chest X-ray."):
    # 1. Process Image
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        # Add batch dimension and match the vision encoder's dtype (e.g. float16)
        vision_dtype = next(model.vision_encoder.parameters()).dtype
        img_tensor = img_transforms(image).unsqueeze(0).to(device, dtype=vision_dtype)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # 2. Get Image Embeddings
    with torch.no_grad():
        image_features = model.vision_encoder(img_tensor)
        projected_features = model.projector(image_features) # Shape: (1, 256, 4096)
        
    # 3. Prepare Text Prompt
    system_prompt = "You are an expert radiologist. Analyze the chest X-ray and provide a detailed report."
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<image>\n{prompt}"}
    ]
    
    prompt_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    # Split prompt around the <image> token placeholder
    parts = prompt_text.split("<image>")
    if len(parts) != 2:
        raise ValueError("Prompt must contain exactly one <image> token placeholder.")
        
    before_img_tokens = tokenizer(parts[0], return_tensors="pt").input_ids.to(device)
    after_img_tokens = tokenizer(parts[1], return_tensors="pt").input_ids.to(device)
    
    # 4. Construct Full Embeddings
    with torch.no_grad():
        before_embeds = model.llm.get_input_embeddings()(before_img_tokens)
        after_embeds = model.llm.get_input_embeddings()(after_img_tokens)
        
        # Concatenate: [Text before] + [Image Features] + [Text after]
        inputs_embeds = torch.cat([before_embeds, projected_features, after_embeds], dim=1)
        
        # 5. Generate Response
        print("\nGenerating report...")
        outputs = model.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=300,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a Chest X-Ray")
    parser.add_argument("--image", type=str, required=True, help="Path to the chest X-ray image (PNG/JPG)")
    parser.add_argument("--prompt", type=str, default="What are the main clinical findings in this radiograph?", help="Question to ask the model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Could not find image at {args.image}")
        exit(1)
        
    print("Initializing OmniMed Inference Engine...")
    model, tokenizer, device = load_inference_model()
    
    print(f"\nAnalyzing: {args.image}")
    report = generate_report(model, tokenizer, device, args.image, args.prompt)
    
    print("\n" + "="*50)
    print("🩺 OMNIMED REPORT")
    print("="*50)
    print(report)
    print("="*50)
