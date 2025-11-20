"""
Gradio App for Adversarial Robustness and OOD Detection Demo
Deployed on Hugging Face Spaces
"""
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load pre-trained model from Hugging Face
MODEL_NAME = "textattack/distilbert-base-uncased-ag-news"
print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully!")

# Import adversarial attacker
from src.adversarial import AdversarialAttacker
print("Loading adversarial attacker...")
attacker = AdversarialAttacker(model_name=MODEL_NAME)
print("All components loaded.")

def analyze_text(text, threshold=0.5):
    """Analyze input text for OOD detection and generate adversarial example"""
    if not text.strip():
        return "Please enter some text.", "", "", ""
    
    # Tokenize and get prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = torch.max(probs).item()
        prediction = torch.argmax(probs, dim=-1).item()
    
    # OOD Detection (using Maximum Softmax Probability)
    is_ood = confidence < threshold
    
    # Label mapping
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    pred_label = label_map.get(prediction, "Unknown")
    
    # Status
    status = "🚫 OUT-OF-DISTRIBUTION" if is_ood else "✅ IN-DISTRIBUTION"
    ood_info = f"{status}\n\nConfidence: {confidence:.4f}\nPredicted Class: {pred_label}"
    
    # Generate adversarial example
    adv_text = ""
    adv_status = ""
    if not is_ood:
        try:
            adv_result, attack_status = attacker.generate_adversarial_example(text, prediction)
            if attack_status == "Success":
                adv_text = adv_result
                adv_status = "⚔️ Attack Successful! Model prediction flipped."
            else:
                adv_status = f"Attack Status: {attack_status}"
        except Exception as e:
            adv_status = f"Attack failed: {str(e)}"
    else:
        adv_status = "Skipped (input is OOD)"
    
    return ood_info, adv_text, adv_status, f"Threshold: {threshold}"

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🛡️ AI Trust & Safety Demo
    
    Test **Out-of-Distribution Detection** and **Adversarial Robustness** for NLP models.
    
    This demo uses a DistilBERT model trained on AG News (World, Sports, Business, Sci/Tech) to demonstrate:
    - **OOD Detection**: Identifies when inputs fall outside the training distribution
    - **Adversarial Attacks**: Generates perturbed text that fools the model
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to analyze...",
                lines=3
            )
            threshold_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.5,
                step=0.1,
                label="OOD Threshold (lower = stricter)"
            )
            analyze_btn = gr.Button("Analyze Input", variant="primary")
        
        with gr.Column():
            ood_output = gr.Textbox(label="OOD Status", lines=4)
            threshold_info = gr.Textbox(label="Settings", lines=1)
    
    gr.Markdown("## ⚔️ Adversarial Attack")
    
    with gr.Row():
        adv_text_output = gr.Textbox(label="Adversarial Example", lines=2)
        adv_status_output = gr.Textbox(label="Attack Status", lines=2)
    
    # Examples
    gr.Examples(
        examples=[
            ["The stock market crashed today due to economic concerns.", 0.5],
            ["The football team won the championship game yesterday.", 0.5],
            ["This movie was absolutely terrible and boring.", 0.5],
            ["Scientists discovered a new exoplanet in a distant galaxy.", 0.5],
        ],
        inputs=[input_text, threshold_slider],
        label="Try these examples"
    )
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_text,
        inputs=[input_text, threshold_slider],
        outputs=[ood_output, adv_text_output, adv_status_output, threshold_info]
    )
    
    gr.Markdown("""
    ---
    **Note**: Adversarial attacks may take 10-30 seconds to generate. The model is trained on AG News dataset.
    
    **Technologies**: DistilBERT • PyTorch • TextAttack • Gradio
    """)

if __name__ == "__main__":
    demo.launch()
