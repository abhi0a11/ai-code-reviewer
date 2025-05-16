from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import wandb
import torch
wandb.login()

# Create a CodeT5 model class for better reusability
class CodeT5Model:
    def __init__(self, model_name="Salesforce/codet5-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True)
        
    def review_and_correct_code(self, code):
        inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs)
        corrected_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected_code
    
    def preprocess_function(self, examples):
        inputs = [ex for ex in examples["code"]]
        targets = [ex for ex in examples["corrected_code"]]
        # Reducing max length from 512 to 256 to save memory
        model_inputs = self.tokenizer(inputs, max_length=256, truncation=True, padding="max_length")
        # Use the text_target parameter instead of as_target_tokenizer
        labels = self.tokenizer(text_target=targets, max_length=256, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def fine_tune(self, tokenized_dataset):
        training_args = Seq2SeqTrainingArguments(
            output_dir="./code_review_model",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=3,
            predict_with_generate=True,
            report_to="wandb",
            fp16=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
        )

        trainer.train()
        return trainer
    
    def save_model(self, trainer, path="./fine_tuned_code_review_model"):
        trainer.save_model(path)
        
    def load_fine_tuned_model(self, path="./fine_tuned_code_review_model"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        return self

# Create an instance of the CodeT5Model
code_t5 = CodeT5Model()

# Example usage with a placeholder dataset
from datasets import Dataset

# Create a small example dataset
data = {
     "code": [
        "def a_function(x, y)\n  return x * y",
        
        "for i in range(5)\n  print(i)",
        
        "print('Hello, World!'\n print('Welcome')",
        
        "def my_function()\n  print('Hello')\nmy_function()",
        
        "x = [1, 2, 3]\ny = [4, 5, 6]\nprint(x + y)",
        
        "if x = 7:\n  print('x is 7')",
        
        "def factorial(n):\n  if n <= 1\n    return 1\n  else\n    return n * factorial(n-1)",
        
        "def greeting(name):\n  prin('Hello, ' + name)",
        
        "for item in ['a', 'b', 'c']\n print(item)",
        
        "name = 'Alice'\nif name.isupper():\n print('Name is uppercase')",
        'for item in ["apple", "banana", "cherry"]:\n print(item)',
        
    ],
    "corrected_code": [
        "def a_function(x, y):\n \treturn x * y",
        
        "for i in range(5):\n \tprint(i)",
        
        "print('Hello, World!')\nprint('Welcome')",
        
        "def my_function():\n \tprint('Hello')\nmy_function()",
        
        "x = [1, 2, 3]\ny = [4, 5, 6]\nprint(x + y)  # This should actually be fixed logically, if you want concatenation, you'd use x += y",
        
        "if x == 7:\n \tprint('x is 7')",
        
        "def factorial(n):\n \tif n <= 1:\n \t\treturn 1\n \telse:\n \t\treturn n * factorial(n-1)",
        
        "def greeting(name):\n \tprint('Hello, ' + name)",
        
        "item='d'\nfor item in ['a', 'b', 'c']:\n \tprint(item)",
        
        # Correcting Example 10
        "name = 'Alice'\nif name.upper():\n \tprint('Name is uppercase')",
        
        # Continue creating corrected examples...
        'item = "apple"\nif item in ["apple", "banana","cherry"]:\n \tprint(item)\nelse:\n \tprint("apple")',
    ],
}
dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2)  # Split into train and validation

# Use the instance methods for preprocessing and fine-tuning
tokenized_dataset = dataset.map(code_t5.preprocess_function, batched=True)
trainer = code_t5.fine_tune(tokenized_dataset)
code_t5.save_model(trainer)

# Example of how to use the model for inference
example_code = "def say_hello(name)\n  print('Hello, ' + name)"
corrected_code = code_t5.review_and_correct_code(example_code)
print("Original code:", example_code)
print("Corrected code:", corrected_code)
