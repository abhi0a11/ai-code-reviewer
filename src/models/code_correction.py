"""
Code correction model using CodeT5.
"""
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model(model_path=None):
    """
    Load the code correction model.
    
    Args:
        model_path: Path to the fine-tuned model. If None, uses the base CodeT5 model.
    
    Returns:
        tuple: (model, tokenizer)
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    else:
        print("Loading base CodeT5 model")
        model_name = "Salesforce/codet5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return model, tokenizer

def review_code(code_to_review, model_path=None, max_length=512):
    """
    Review and correct code.
    
    Args:
        code_to_review: Code snippet with potential errors
        model_path: Path to the fine-tuned model. If None, uses the base CodeT5 model.
        max_length: Maximum length of the generated code
    
    Returns:
        str: Corrected code
    """
    model, tokenizer = load_model(model_path)
    
    # Tokenize and generate corrected code
    inputs = tokenizer(code_to_review, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    corrected_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return corrected_code

def batch_review_code(code_snippets, model_path=None, max_length=512):
    """
    Review and correct multiple code snippets in batch.
    
    Args:
        code_snippets: List of code snippets with potential errors
        model_path: Path to the fine-tuned model. If None, uses the base CodeT5 model.
        max_length: Maximum length of the generated code
    
    Returns:
        list: List of corrected code snippets
    """
    model, tokenizer = load_model(model_path)
    
    # Tokenize and generate corrected code
    inputs = tokenizer(code_snippets, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    corrected_code = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    return corrected_code
