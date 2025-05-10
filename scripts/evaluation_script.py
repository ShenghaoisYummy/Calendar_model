#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calendar Event Model Evaluation Script
======================================

This script evaluates a fine-tuned model on calendar event generation.
"""

import os
import argparse
import json
import wandb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import (
    load_data,
    get_model_predictions,
    process_model_outputs,
    CalendarEventEvaluator
)
from src.fine_tune import setup_model_and_tokenizer

# Default system prompt for the model
DEFAULT_SYSTEM_PROMPT = """You are a scheduling assistant.
Read the user's request and output **only** a JSON object
with the following exact keys, in this order (no extra text before/after):
'title', 'type', 'description', 'date', 'startTime', 'endTime',
'location', 'isAllDay', 'response'
• 'date', 'startTime', 'endTime' must be RFC3339/ISO-8601 strings.
• 'type' ∈ {add, delete, edit, query, chat}. For creating a new event use "add".
• 'isAllDay' = 1 if the request is an all-day event, else 0.
• 'response' is how you would politely confirm the action to the user.
Output NOTHING except that JSON object."""

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned calendar event model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--wandb_project", type=str, default="calendar-model-evaluation", help="W&B project name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples to evaluate")
    parser.add_argument("--system_prompt", type=str, default=None, help="Custom system prompt (or 'none' to disable)")
    args = parser.parse_args()
    
    # Initialize W&B
    wandb.init(project=args.wandb_project, config=vars(args))
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_data = load_data(args.test_data)
    
    # Limit samples if specified
    if args.max_samples and args.max_samples < len(test_data):
        test_data = test_data[:args.max_samples]
    
    # Extract prompts from test data
    prompts = [item.get("text", "") for item in test_data]
    references = test_data
    
    # Determine system prompt
    system_prompt = None
    if args.system_prompt and args.system_prompt.lower() != "none":
        system_prompt = args.system_prompt
    elif args.system_prompt is None:  # Use default if not specified
        system_prompt = DEFAULT_SYSTEM_PROMPT
        
    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt will be used")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model, tokenizer = setup_model_and_tokenizer(args.model_path)
    
    # Generate predictions
    print("Generating predictions...")
    raw_outputs = get_model_predictions(
        model, 
        tokenizer, 
        prompts, 
        system_prompt=system_prompt, 
        batch_size=args.batch_size
    )
    predictions = process_model_outputs(raw_outputs)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    evaluator = CalendarEventEvaluator()
    results = evaluator.evaluate_batch(references, predictions)
    
    # Save detailed results
    output_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log results to W&B
    # Log overall metrics
    wandb.log({
        "overall_score": results["overall_score"]["mean"],
        "num_samples": results["num_events"]
    })
    
    # Log field-specific metrics
    for field, metrics in results["field_scores"].items():
        wandb.log({f"{field}_score": metrics["mean"]})
    
    # Log confusion matrix for event type if available
    if "type_confusion_matrix" in results:
        wandb.log({"type_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=results["type_confusion_matrix"]["true_labels"],
            preds=results["type_confusion_matrix"]["pred_labels"],
            class_names=results["type_confusion_matrix"]["class_names"]
        )})
    
    print(f"Evaluation complete. Results saved to {output_file}")
    print(f"Overall score: {results['overall_score']['mean']:.4f}")

if __name__ == "__main__":
    main()