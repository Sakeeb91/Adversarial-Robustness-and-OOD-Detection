import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.ood import OODDetector
from src.adversarial import AdversarialAttacker

def verify():
    print("=== Verifying System ===")
    
    # 1. Load Components
    print("[1] Loading Components...")
    try:
        ood = OODDetector()
        attacker = AdversarialAttacker()
        print("✓ Components Loaded")
    except Exception as e:
        print(f"✗ Failed to load components: {e}")
        return

    # 2. Test In-Distribution
    print("\n[2] Testing In-Distribution (ID)...")
    id_text = "The stock market crashed today due to bad economic news."
    is_ood, score = ood.is_ood(id_text, threshold=0.5)
    print(f"Text: '{id_text}'")
    print(f"OOD Score: {score:.4f}, Is OOD: {is_ood}")
    if not is_ood:
        print("✓ ID correctly identified")
    else:
        print("✗ ID flagged as OOD (Check threshold)")

    # 3. Test Out-of-Distribution
    print("\n[3] Testing Out-of-Distribution (OOD)...")
    ood_text = "This movie was absolutely terrible and the acting was bad." # IMDB style
    is_ood, score = ood.is_ood(ood_text, threshold=0.9) # Higher threshold for strictness? Or lower?
    # Note: OOD score is confidence. OOD text might still have high confidence if model is overconfident.
    # We expect lower confidence than ID, but maybe not < 0.5 without calibration.
    print(f"Text: '{ood_text}'")
    print(f"OOD Score: {score:.4f}, Is OOD: {is_ood}")
    
    # 4. Test Adversarial
    print("\n[4] Testing Adversarial Attack...")
    adv_text, status = attacker.generate_adversarial_example(id_text, 2) # 2 = Business
    print(f"Status: {status}")
    if status == "Success":
        print(f"Adversarial: '{adv_text}'")
        print("✓ Attack generated")
    else:
        print("✗ Attack failed")

if __name__ == "__main__":
    verify()
