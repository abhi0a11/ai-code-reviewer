from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def review_code(code_to_review):
    # Load the model
    model_name = "Salesforce/codet5-base"  # Or use your fine-tuned model path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Tokenize and generate corrected code
    inputs = tokenizer(code_to_review, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(**inputs, max_length=512)
    corrected_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return corrected_code

# Example usage
if __name__ == "__main__":
    # Example code with errors
    code_with_errors = """
def calculate_sum(a, b)
    return a + b

for i in range(5)
    print(i)
    """
    
    corrected = review_code(code_with_errors)
    
    print("Original code:")
    print(code_with_errors)
    print("\nCorrected code:")
    print(corrected) 