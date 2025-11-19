import torch
import numpy as np
from src.model import load_trained_model

class OODDetector:
    def __init__(self, model_path='./models/base_model'):
        self.model, self.tokenizer = load_trained_model(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()[0]

    def get_ood_score(self, text):
        """
        Returns the OOD score. Higher score means more likely to be ID (confidence).
        So, lower score means OOD.
        Using MSP (Maximum Softmax Probability).
        """
        probs = self.predict(text)
        return np.max(probs)

    def is_ood(self, text, threshold=0.5):
        """
        Returns True if OOD, False if ID.
        Threshold: if max prob < threshold, then OOD.
        """
        score = self.get_ood_score(text)
        return score < threshold, score
