import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.utils import load_data, get_tokenizer
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, predictions)}

def train_model(model_name='distilbert-base-uncased', output_dir='./models/base_model', epochs=1):
    """
    Train a baseline model on AG News.
    """
    tokenizer = get_tokenizer(model_name)
    
    # Load AG News (In-Distribution)
    dataset = load_data('ag_news', split='train', sample_size=2000) # Small sample for demo speed
    test_dataset = load_data('ag_news', split='test', sample_size=500)
    
    def tokenize(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)
    
    tokenized_train = dataset.map(tokenize, batched=True)
    tokenized_test = test_dataset.map(tokenize, batched=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="no", # Save space
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    return model, tokenizer

def load_trained_model(model_path='./models/base_model'):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = get_tokenizer(model_path)
    return model, tokenizer

if __name__ == "__main__":
    train_model()
