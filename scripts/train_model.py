"""
Script to train the code review model
"""
import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.data.dataset import load_dataset_from_file, prepare_dataset_for_training
from src.training.trainer import get_training_args, create_trainer, train_model, evaluate_model
import wandb
from main import code_t5
from datasets import load_dataset, Dataset
import torch

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train a code review model")
    parser.add_argument("--model_name", type=str, default="Salesforce/codet5-base", help="Base model to fine-tune")
    parser.add_argument("--data_file", type=str, default=None, help="Path to the dataset file")
    parser.add_argument("--output_dir", type=str, default="./code_review_model", help="Directory to save the model checkpoints")
    parser.add_argument("--final_model_dir", type=str, default="./fine_tuned_code_review_model", help="Directory to save the final model")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for the optimizer")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of the tokenized inputs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    return parser.parse_args()

def main():
    """
    Main function to train the model
    """
    args = parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="code-review-model", config=vars(args))
    
    # Load the model and tokenizer
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, low_cpu_mem_usage=True)
    
    # Load and prepare dataset
    print(f"Loading dataset from {args.data_file if args.data_file else 'example data'}...")
    dataset_dict = load_dataset_from_file(args.data_file) if args.data_file else None
    if dataset_dict is None:
        from src.data.dataset import create_example_dataset
        dataset_dict = create_example_dataset()
    
    print(f"Preparing dataset for training...")
    tokenized_dataset = prepare_dataset_for_training(dataset_dict, tokenizer, max_length=args.max_length)
    
    # Configure training arguments
    training_args = get_training_args(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        report_to="wandb" if args.use_wandb else "none"
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = create_trainer(model, tokenizer, tokenized_dataset["train"], tokenized_dataset["test"], training_args)
    
    # Train the model
    print("Training the model...")
    train_metrics = train_model(trainer, save_path=args.final_model_dir)
    print(f"Training metrics: {train_metrics}")
    
    # Evaluate the model
    print("Evaluating the model...")
    eval_metrics = evaluate_model(trainer)
    print(f"Evaluation metrics: {eval_metrics}")
    
    # Finish wandb run if used
    if args.use_wandb:
        wandb.finish()
    
    print(f"Model saved to {args.final_model_dir}")

def train_model():
    # Load your training dataset or use the example dataset
    # You can replace this with loading your actual dataset
    data = {
        "code": [
            "def a_function(x, y)\n  return x * y",
            "for i in range(5)\n  print(i)",
            # ... more examples ...
        ],
        "corrected_code": [
            "def a_function(x, y):\n \treturn x * y",
            "for i in range(5):\n \tprint(i)",
            # ... more examples ...
        ],
    }
    
    # Create dataset
    dataset = Dataset.from_dict(data)
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(code_t5.preprocess_function, batched=True)
    
    # Fine-tune the model
    trainer = code_t5.fine_tune(tokenized_dataset)
    
    # Save the model
    code_t5.save_model(trainer)
    
    return trainer

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train the model
    trainer = train_model()
    
    # Print training results
    print("Training completed.")
    print("Evaluation results:")
    
    # Get evaluation results and print each on a new line
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value}")
