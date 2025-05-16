"""
Dataset handling utilities for code review model
"""
from datasets import Dataset, DatasetDict
import os
import json
from .preprocessing import clean_code

def create_example_dataset():
    """
    Create an example dataset for code review.
    
    Returns:
        DatasetDict: A dataset with train and test splits
    """
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
    return DatasetDict({
        "train": dataset.select(range(0, 8)),
        "test": dataset.select(range(8, 10))
    })

def load_dataset_from_file(file_path):
    """
    Load dataset from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing the dataset
        
    Returns:
        DatasetDict: A dataset with train and test splits
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist. Creating example dataset.")
        return create_example_dataset()
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Clean the code samples
    data["code"] = [clean_code(code) for code in data["code"]]
    data["corrected_code"] = [clean_code(code) for code in data["corrected_code"]]
    
    dataset = Dataset.from_dict(data)
    # Split 80% train, 20% test
    dataset_dict = dataset.train_test_split(test_size=0.2)
    
    return dataset_dict

def prepare_dataset_for_training(dataset_dict, tokenizer, max_length=256):
    """
    Prepare dataset for training by tokenizing the inputs and targets.
    
    Args:
        dataset_dict: DatasetDict containing train and test splits
        tokenizer: Tokenizer for the model
        max_length: Maximum length of the tokenized inputs
        
    Returns:
        DatasetDict: Tokenized dataset
    """
    def preprocess_function(examples):
        inputs = examples["code"]
        targets = examples["corrected_code"]
        
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
        labels = tokenizer(text_target=targets, max_length=max_length, truncation=True, padding="max_length")
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=["code", "corrected_code"]
    )
    
    return tokenized_dataset
