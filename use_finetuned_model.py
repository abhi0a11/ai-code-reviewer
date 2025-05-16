from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def review_with_finetuned_model(code_to_review):
    # Load the fine-tuned model or fall back to base model
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
    
    # Tokenize and generate corrected code
    inputs = tokenizer(code_to_review, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=256)
    corrected_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Compare and show differences
    print("Original code:")
    print(code_to_review)
    print("\nCorrected code:")
    print(corrected_code)
    
    return corrected_code

# Example usage
if __name__ == "__main__":
    # Example code with errors
    code_with_errors = """
# Function with missing colon and indentation issues
def calculate_factorial(n)
  if n <= 1
    return 1
  else
    return n * calculate_factorial(n-1)
    
# Loop with missing colon
for item in ["apple", "banana", "cherry"]
print(item)  # Incorrect indentation
    """
    
    review_with_finetuned_model(code_with_errors) 