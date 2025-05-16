"""
Interactive code review and correction tool.
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

def review_and_correct_code(code, tokenizer, model):
    """
    Review and correct code using the model
    """
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    corrected_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_code

def main():
    """
    Main interactive loop
    """
    print("Loading model... (this may take a moment)")
    tokenizer, model = load_model()
    print("\nCode Review Tool")
    print("Type 'exit' when done, or 'file' to review code from a file")
    
    while True:
        print("\nEnter Python code to review (type 'exit' to quit, 'file' to review from file):")
        
        # Collect multiline input
        code_lines = []
        while True:
            line = input()
            if line.strip().lower() == 'exit':
                return
            if line.strip().lower() == 'done':
                break
            if line.strip().lower() == 'file':
                filepath = input("Enter path to the Python file: ")
                try:
                    with open(filepath, 'r') as f:
                        code_lines = f.readlines()
                    break
                except Exception as e:
                    print(f"Error reading file: {e}")
                    continue
            code_lines.append(line)
        
        if not code_lines:
            continue
            
        code = "\n".join(code_lines)
        
        print("\nReviewing code...")
        corrected_code = review_and_correct_code(code, tokenizer, model)
        
        print("\nOriginal Code:")
        print(code)
        print("\nCorrected Code:")
        print(corrected_code)

if __name__ == "__main__":
    main() 