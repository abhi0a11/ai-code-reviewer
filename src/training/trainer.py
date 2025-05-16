"""
Training utilities for the code correction model
"""
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
import torch
from .metrics import compute_metrics

def get_training_args(
    output_dir="./code_review_model",
    learning_rate=2e-5,
    batch_size=2,
    num_epochs=3,
    weight_decay=0.01,
    report_to="wandb"
):
    """
    Get training arguments for the model.
    
    Args:
        output_dir: Directory to save the model checkpoints
        learning_rate: Learning rate for the optimizer
        batch_size: Batch size for training and evaluation
        num_epochs: Number of training epochs
        weight_decay: Weight decay for the optimizer
        report_to: Reporting platform (wandb, tensorboard, etc.)
        
    Returns:
        Seq2SeqTrainingArguments: Training arguments
    """
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        weight_decay=weight_decay,
        save_total_limit=3,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        report_to=report_to,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        generation_max_length=256,
    )

def create_trainer(model, tokenizer, train_dataset, eval_dataset, training_args=None):
    """
    Create a trainer for the model.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer for the model
        train_dataset: The training dataset
        eval_dataset: The evaluation dataset
        training_args: Training arguments
        
    Returns:
        Seq2SeqTrainer: Trainer for the model
    """
    if training_args is None:
        training_args = get_training_args()
    
    # Import needed for compute_metrics
    from .metrics import compute_metrics
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # Remove tokenizer parameter to avoid deprecation warning
        compute_metrics=compute_metrics
    )
    
    return trainer

def train_model(trainer, save_path="./fine_tuned_code_review_model"):
    """
    Train the model and save it.
    
    Args:
        trainer: The trainer to use
        save_path: Path to save the model
        
    Returns:
        dict: Training metrics
    """
    # Train the model
    train_result = trainer.train()
    metrics = train_result.metrics
    
    # Save the model
    trainer.save_model(save_path)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    return metrics

def evaluate_model(trainer):
    """
    Evaluate the model.
    
    Args:
        trainer: The trainer to use
        
    Returns:
        dict: Evaluation metrics
    """
    metrics = trainer.evaluate(
        max_length=256,
        num_beams=4,
        metric_key_prefix="eval"
    )
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    return metrics
