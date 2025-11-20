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

# Create Gradio Interface with Cybersecurity Dark Theme
theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_border_color="#334155",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#475569",
    input_border_width="1px",
    button_primary_background_fill="#06b6d4",
    button_primary_background_fill_hover="#0891b2",
    button_primary_text_color="#ffffff",
    block_label_text_color="#f1f5f9",
    block_title_text_color="#ffffff",
    body_text_color="#e2e8f0",
    body_text_color_subdued="#cbd5e1",
)

with gr.Blocks(theme=theme, css="""
    .gradio-container {
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #ffffff !important;
        font-weight: 700;
    }
    h2 {
        color: #f1f5f9 !important;
    }
    .contain {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    /* Fix input text visibility */
    input, textarea {
        color: #f1f5f9 !important;
        background: #0f172a !important;
    }
    /* Fix example text visibility */
    .examples table td {
        color: #e2e8f0 !important;
        background: #1e293b !important;
    }
    .examples table th {
        color: #f1f5f9 !important;
        background: #334155 !important;
    }
    /* Fix label text */
    label {
        color: #f1f5f9 !important;
    }
    /* Fix output text boxes */
    .output-class {
        color: #e2e8f0 !important;
    }
""") as demo:
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
