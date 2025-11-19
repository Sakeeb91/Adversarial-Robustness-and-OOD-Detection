import gradio as gr
from src.ood import OODDetector
from src.adversarial import AdversarialAttacker
import numpy as np

# Initialize models globally to avoid reloading
print("Loading models...")
ood_detector = OODDetector()
attacker = AdversarialAttacker()
print("Models loaded.")

labels = ["World", "Sports", "Business", "Sci/Tech"]

def analyze_text(text):
    # 1. OOD Check
    is_ood, score = ood_detector.is_ood(text, threshold=0.5)
    
    # 2. Prediction
    probs = ood_detector.predict(text)
    pred_idx = np.argmax(probs)
    pred_label = labels[pred_idx]
    confidence = probs[pred_idx]
    
    ood_status = "🔴 Out-of-Distribution" if is_ood else "🟢 In-Distribution"
    ood_details = f"Confidence Score: {score:.4f} (Threshold: 0.5)"
    
    return ood_status, ood_details, f"{pred_label} ({confidence:.4f})", pred_idx

def generate_attack(text, pred_idx):
    if not text:
        return "No text provided.", "", ""
    
    adv_text, status = attacker.generate_adversarial_example(text, int(pred_idx))
    
    if status == "Success":
        # Analyze adversarial text
        probs = ood_detector.predict(adv_text)
        adv_pred_idx = np.argmax(probs)
        adv_pred_label = labels[adv_pred_idx]
        confidence = probs[adv_pred_idx]
        
        result_msg = "✅ Attack Successful! Label flipped."
        return adv_text, f"{adv_pred_label} ({confidence:.4f})", result_msg
    else:
        return text, "N/A", "❌ Attack Failed or Error."

with gr.Blocks(title="Adversarial Robustness & OOD Demo") as demo:
    gr.Markdown("# 🛡️ AI Trust & Safety Demo")
    gr.Markdown("Demonstrating **Out-of-Distribution Detection** and **Adversarial Robustness**.")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", placeholder="Enter news headline or text...", lines=3)
            analyze_btn = gr.Button("Analyze Input", variant="primary")
        
        with gr.Column():
            ood_status_output = gr.Label(label="OOD Status")
            ood_details_output = gr.Textbox(label="OOD Details")
            prediction_output = gr.Textbox(label="Model Prediction")
            # Hidden state to store prediction index for attack
            pred_idx_state = gr.State()

    gr.Markdown("---")
    gr.Markdown("### ⚔️ Adversarial Attack")
    
    attack_btn = gr.Button("Generate Adversarial Example", variant="stop")
    
    with gr.Row():
        with gr.Column():
            adv_text_output = gr.Textbox(label="Adversarial Text", lines=3, interactive=False)
        with gr.Column():
            adv_prediction_output = gr.Textbox(label="Adversarial Prediction")
            attack_result_output = gr.Label(label="Attack Result")

    # Event Handlers
    analyze_btn.click(
        analyze_text,
        inputs=[input_text],
        outputs=[ood_status_output, ood_details_output, prediction_output, pred_idx_state]
    )
    
    attack_btn.click(
        generate_attack,
        inputs=[input_text, pred_idx_state],
        outputs=[adv_text_output, adv_prediction_output, attack_result_output]
    )

    gr.Markdown("""
    **Examples:**
    * *ID (Business)*: "The stock market crashed today due to bad economic news."
    * *ID (Sports)*: "The team won the championship game yesterday."
    * *OOD (Movie Review)*: "This movie was absolutely terrible and the acting was bad."
    """)

if __name__ == "__main__":
    demo.launch(share=False)
