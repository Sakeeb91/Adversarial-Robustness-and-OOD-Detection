import argparse
from src.ood import OODDetector
from src.adversarial import AdversarialAttacker
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness and OOD Detection Demo")
    parser.add_argument("--text", type=str, help="Input text to classify")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    args = parser.parse_args()

    print("Loading models...")
    ood_detector = OODDetector()
    attacker = AdversarialAttacker()
    print("Models loaded.")

    labels = ["World", "Sports", "Business", "Sci/Tech"]

    def process_input(text):
        print(f"\nInput: '{text}'")
        
        # 1. OOD Check
        is_ood, score = ood_detector.is_ood(text, threshold=0.5) # Threshold might need tuning
        print(f"OOD Score (Confidence): {score:.4f}")
        if is_ood:
            print("Status: OUT-OF-DISTRIBUTION (Uncertain)")
        else:
            print("Status: IN-DISTRIBUTION")

        # 2. Prediction
        probs = ood_detector.predict(text)
        pred_idx = np.argmax(probs)
        pred_label = labels[pred_idx]
        print(f"Prediction: {pred_label} ({probs[pred_idx]:.4f})")

        # 3. Adversarial Attack
        if not is_ood:
            print("\nGenerating Adversarial Example...")
            adv_text, status = attacker.generate_adversarial_example(text, int(pred_idx))
            if status == "Success":
                print(f"Adversarial Text: '{adv_text}'")
                adv_probs = ood_detector.predict(adv_text)
                adv_pred_idx = np.argmax(adv_probs)
                adv_pred_label = labels[adv_pred_idx]
                print(f"Adversarial Prediction: {adv_pred_label} ({adv_probs[adv_pred_idx]:.4f})")
                if adv_pred_idx != pred_idx:
                    print("Attack Successful: Label flipped!")
                else:
                    print("Attack Failed: Label unchanged.")
            else:
                print("Could not generate adversarial example.")

    if args.interactive:
        print("\nEnter text to analyze (type 'exit' to quit):")
        while True:
            text = input("> ")
            if text.lower() == 'exit':
                break
            process_input(text)
    elif args.text:
        process_input(args.text)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
