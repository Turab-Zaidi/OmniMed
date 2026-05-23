import gradio as gr
import torch
from PIL import Image
import sys
import os

# Ensure src module is discoverable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.inference import load_inference_model, generate_report
from torchvision import transforms

print("Initializing OmniMed Vision-Language Model...")
model, tokenizer, device = load_inference_model()
print("Model loaded successfully!")

# Define the exact same image transforms used in training/inference
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
])

def analyze_xray(image):
    if image is None:
        return "Please upload an X-ray image."
    
    try:
        # Preprocess the image
        img_tensor = img_transforms(image).unsqueeze(0).to(device, dtype=torch.float16)
        
        # Generate the medical report
        report = generate_report(
            model=model, 
            tokenizer=tokenizer, 
            device=device, 
            image_path=image, 
            prompt="You are an expert radiologist. Analyze the chest X-ray and provide a detailed report."
        )
        return report
    except Exception as e:
        return f"An error occurred during analysis: {str(e)}"

# Create a sleek, modern Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto;">
            <h1>🩺 OmniMed Vision-Language Model</h1>
            <p style="font-size: 16px;">
                A multi-modal AI designed to analyze Chest X-Rays and generate professional radiological reports.<br>
                Powered by <b>Llama-3 (8B)</b> and <b>BiomedCLIP</b>.
            </p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Chest X-Ray", height=400)
            submit_btn = gr.Button("🔍 Analyze X-Ray", variant="primary")
            
            gr.Examples(
                examples=[["assets/sample_xray.png"]],
                inputs=image_input,
                label="Sample X-Rays (Click to test)"
            )
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Generated Medical Report", 
                lines=18, 
                show_copy_button=True
            )
            
    submit_btn.click(
        fn=analyze_xray, 
        inputs=[image_input], 
        outputs=[output_text],
        api_name="analyze"
    )

if __name__ == "__main__":
    # Launch the Gradio web server
    demo.launch(server_name="0.0.0.0", share=True)
