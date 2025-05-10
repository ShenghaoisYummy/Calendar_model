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
from src.constants import DEFAULT_SYSTEM_PROMPT, EVALUATION_DATA_PATH, MODEL_PATH

def print_detailed_results(references, predictions, results):
    """Print detailed results for each prediction"""
    print("\nDetailed Results:")
    print("=" * 80)
    
    for i, (ref, pred, event_result) in enumerate(zip(references, predictions, results['event_results'])):
        print(f"\nSample {i+1}:")
        print("-" * 40)
        
        # Print reference
        print("Reference:")
        for field, value in ref.items():
            if field not in ['_id', '_source']:  # Skip internal fields
                print(f"  {field}: {value}")
        
        # Print prediction
        print("\nPrediction:")
        for field, value in pred.items():
            if field not in ['_id', '_source']:  # Skip internal fields
                print(f"  {field}: {value}")
        
        # Print evaluation scores with all details
        print("\nEvaluation Scores:")
        for field, field_result in event_result['fields'].items():
            print(f"  {field}:")
            for metric, score in field_result.items():
                if isinstance(score, (int, float)):
                    print(f"    - {metric}: {score:.3f}")
                else:
                    print(f"    - {metric}: {score}")
        
        print(f"\nOverall Score: {event_result['overall_score']:.3f}")
        print("=" * 80)
    
    # Print aggregate statistics
    print("\nAggregate Statistics:")
    print("-" * 40)
    print(f"Number of Samples: {results['num_events']}")
    print(f"Overall Score: {results['overall_score']['mean']:.4f} ± {results['overall_score']['std']:.4f}")
    print(f"Completeness: {results['completeness']['mean']:.4f} ± {results['completeness']['std']:.4f}")
    
    # Print field scores
    print("\nField Scores:")
    for field, metrics in results['field_scores'].items():
        if field == 'intent':
            print(f"  {field}: {metrics['mean']:.4f} (Accuracy) (n={metrics['count']})")
        else:
            print(f"  {field}: {metrics['mean']:.4f} ± {metrics['std']:.4f} (n={metrics['count']})")
    
    # Print confusion matrix if available
    if 'intent_confusion_matrix' in results:
        print("\nIntent Confusion Matrix:")
        cm = results['intent_confusion_matrix']['matrix']
        class_names = results['intent_confusion_matrix']['class_names']
        
        # Print header
        header = "    "
        for name in class_names:
            header += f"{name[:10]:>10} "
        print(header)
        
        # Print rows
        for i, name in enumerate(class_names):
            row = f"{name[:10]:<10} "
            for j in range(len(cm[i])):
                row += f"{cm[i][j]:>10} "
            print(row)
        
        # Print metrics
        intent_accuracy = results['intent_confusion_matrix']['accuracy']
        print(f"\nIntent Classification Accuracy: {intent_accuracy:.4f} ** (Used for intent field score)")
        print(f"Intent Classification F1 Score: {results['intent_confusion_matrix']['f1_score']:.4f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned calendar event model")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the fine-tuned model")
    parser.add_argument("--test_data", type=str, default=EVALUATION_DATA_PATH, help="Path to test data file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--wandb_project", type=str, default="calendar-model-evaluation", help="W&B project name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples to evaluate")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="Custom system prompt (or 'none' to disable)")
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