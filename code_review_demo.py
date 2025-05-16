"""
Code review and correction demo using CodeT5
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def load_model():
    """
    Load the model, first trying the fine-tuned model, then falling back to base model
    """
    model_path = "./fine_tuned_code_review_model"
    base_model = "Salesforce/codet5-base"
    
    # Check if the fine-tuned model exists
    if os.path.exists(model_path) and os.path.isdir(model_path) and any(os.listdir(model_path)):
        print(f"Loading fine-tuned model from {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        except (OSError, ValueError) as e:
            print(f"Error loading fine-tuned model: {e}")
            print(f"Falling back to base model {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    else:
        print(f"Fine-tuned model not found. Using base model {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    
    return tokenizer, model

def review_and_correct_code(code, tokenizer=None, model=None):
    """
    Review and correct code using the model
    """
    # Load model if not provided
    if tokenizer is None or model is None:
        tokenizer, model = load_model()
    
    # Tokenize and generate corrected code
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs)
    corrected_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_code

if __name__ == "__main__":
    # Load the model once to reuse
    tokenizer, model = load_model()
    
    # Example code with errors
    examples = [
        """
def calculate_sum(a, b):
  retur a + b  # Typo here
        """,
        
        """
for i in range(5)
  print(i)
        """,
        
        """
if x = 10:
    print("x is 10")
        """
    ]
    
    # Process each example
    for i, code in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Original Code:\n{code}")
        
        corrected_code = review_and_correct_code(code, tokenizer, model)
        
        print(f"\nCorrected Code:\n{corrected_code}")
        print("-" * 50) 