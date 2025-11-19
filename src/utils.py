import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def load_data(dataset_name, split='train', sample_size=None):
    """
    Load a dataset from Hugging Face.
    """
    dataset = load_dataset(dataset_name, split=split)
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
    return dataset

def get_tokenizer(model_name='distilbert-base-uncased'):
    """
    Load tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples, tokenizer):
    """
    Tokenize text data.
    """
    return tokenizer(examples['text'], padding='max_length', truncation=True)
