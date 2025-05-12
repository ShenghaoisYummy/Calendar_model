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
    get_gpt_predictions,
    process_model_outputs,
    setup_pretrained_model_and_tokenizer,
    CalendarEventEvaluator,
    print_detailed_results
)
from src.fine_tune import setup_model_and_tokenizer
from constants import (
    DEFAULT_SYSTEM_PROMPT, 
    EVALUATION_DATA_PATH, 
    DOWNLOAD_MODEL_PATH,
    ENV_WANDB_API_KEY,
    WANDB_EVAL_PROJECT_NAME
)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned calendar event model")
    parser.add_argument("--download_model_path", type=str, default=DOWNLOAD_MODEL_PATH, help="Path to the fine-tuned model")
    parser.add_argument("--test_data", type=str, default=EVALUATION_DATA_PATH, help="Path to test data file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--wandb_project", type=str, default=WANDB_EVAL_PROJECT_NAME, help="W&B project name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples to evaluate")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt (or 'none' to disable)")
    args = parser.parse_args()
    
    # Check for WANDB API key
    wandb_api_key = os.environ.get(ENV_WANDB_API_KEY)
    if not wandb_api_key:
        print(f"Warning: {ENV_WANDB_API_KEY} environment variable not set. W&B logging may not work properly.")
    
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
    
   
    print(f"Loading model")
    # Load base model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    # # Load pretrained model and tokenizer
    # model, tokenizer = setup_pretrained_model_and_tokenizer()

    
    # Generate predictions
    print("Generating predictions...")
    raw_outputs = get_model_predictions(
        model, 
        tokenizer, 
        prompts, 
        system_prompt=system_prompt, 
        batch_size=args.batch_size
    )
    
    # Print sample of raw outputs for debugging
    print("\nSample of raw outputs:")
    for i in range(min(3, len(raw_outputs))):
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompts[i][:100]}...")
        print(f"Raw output: {raw_outputs[i]}")
        print("-" * 50)
    
    # Process outputs
    print("\nProcessing model outputs...")
    predictions = process_model_outputs(raw_outputs)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    evaluator = CalendarEventEvaluator()
    results = evaluator.evaluate_batch(references, predictions)
    
    # Print detailed results
    print_detailed_results(references, predictions, results)
    
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
    
    # Log confusion matrix for intent field if available
    if "intent_confusion_matrix" in results:
        wandb.log({"intent_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=results["intent_confusion_matrix"]["true_labels"],
            preds=results["intent_confusion_matrix"]["pred_labels"],
            class_names=results["intent_confusion_matrix"]["class_names"]
        )})
    
    print(f"\nEvaluation complete. Results saved to {output_file}")
    print(f"Overall score: {results['overall_score']['mean']:.4f}")

if __name__ == "__main__":
    main()