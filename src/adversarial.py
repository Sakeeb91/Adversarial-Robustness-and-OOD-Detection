import torch
from textattack.attack_recipes import DeepWordBugGao2018
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class AdversarialAttacker:
    def __init__(self, model_name='textattack/distilbert-base-uncased-ag-news'):
        """Initialize with a Hugging Face model name or local path"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model_wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer)
        self.attack = DeepWordBugGao2018.build(self.model_wrapper)

    def generate_adversarial_example(self, text, label):
        """
        Generates an adversarial example for the given text and label.
        Returns the adversarial text and the result.
        """
        from textattack import Attacker
        from textattack.datasets import Dataset
        from textattack.goal_function_results import GoalFunctionResultStatus

        # Create a simple dataset-like object
        dataset = Dataset([(text, label)])
        
        # Use Attacker to run the attack
        attacker = Attacker(self.attack, dataset)
        results = attacker.attack_dataset()
        
        if results:
            result = results[0]
        from textattack.attack_results import SuccessfulAttackResult
        
        if results:
            result = results[0]
            if isinstance(result, SuccessfulAttackResult):
                return result.perturbed_result.attacked_text.text, "Success"
            else:
                return text, "Failed"
        return text, "Error"

if __name__ == "__main__":
    attacker = AdversarialAttacker()
    text = "Sports are great and I love watching football."
    label = 1 # Sports
    adv_text, status = attacker.generate_adversarial_example(text, label)
    print(f"Original: {text}")
    print(f"Adversarial: {adv_text}")
    print(f"Status: {status}")
