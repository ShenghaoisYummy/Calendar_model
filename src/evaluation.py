#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Calendar Event Evaluation Module
===============================

This module provides classes and functions for evaluating calendar event extraction models.
It includes specialized evaluations for different field types:
- Text fields (title, description, response): Rouge-L and BLEU metrics
- Date/time fields: Format validation and value matching
- Type field: Classification accuracy with confusion matrix
- Location field: Custom similarity metrics
"""

import os
import re
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Union, Optional
from datetime import datetime
import dateutil.parser
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import torch
from tqdm import tqdm

# Text similarity metrics
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: rouge_score or nltk not available. Text metrics will be limited.")
    METRICS_AVAILABLE = False

class DateTimeUtils:
    """Utilities for handling date and time formats"""
    
    @staticmethod
    def is_valid_date_format(date_str: str) -> bool:
        """Check if string is a valid ISO date format (YYYY-MM-DD)"""
        if not date_str:
            return False
        try:
            # Convert to string if it's a float
            if isinstance(date_str, float):
                date_str = str(int(date_str))  # Convert float to int then to string
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except (ValueError, TypeError):
            return False
            
    @staticmethod
    def is_valid_iso_datetime(dt_str: str) -> bool:
        """Check if string is a valid ISO datetime format"""
        if not dt_str:
            return False
        try:
            dateutil.parser.isoparse(dt_str)
            return True
        except (ValueError, TypeError):
            return False
            
    @staticmethod
    def extract_date_from_iso(dt_str: str) -> str:
        """Extract just the date part from ISO datetime string"""
        if not dt_str:
            return ""
        try:
            dt = dateutil.parser.isoparse(dt_str)
            return dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return ""
            
    @staticmethod
    def extract_time_from_iso(dt_str: str) -> str:
        """Extract just the time part from ISO datetime string"""
        if not dt_str:
            return ""
        try:
            dt = dateutil.parser.isoparse(dt_str)
            return dt.strftime("%H:%M:%S")
        except (ValueError, TypeError):
            return ""
            
    @staticmethod
    def compare_iso_datetimes(dt1: str, dt2: str) -> bool:
        """Compare two ISO datetime strings for equality (ignoring milliseconds)"""
        if not dt1 or not dt2:
            return False
        try:
            parsed1 = dateutil.parser.isoparse(dt1)
            parsed2 = dateutil.parser.isoparse(dt2)
            # Truncate microseconds for comparison
            parsed1 = parsed1.replace(microsecond=0)
            parsed2 = parsed2.replace(microsecond=0)
            return parsed1 == parsed2
        except (ValueError, TypeError):
            return False
            
    @staticmethod
    def convert_to_iso_format(date_str: str, time_str: str = None) -> str:
        """Convert date and optional time to ISO format"""
        if not date_str:
            return ""
            
        try:
            # Handle different date formats
            date_formats = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%B %d %Y", "%d %B %Y"]
            parsed_date = None
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
                    
            if parsed_date is None:
                return ""
                
            # If time is provided, add it to the datetime
            if time_str:
                time_formats = ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M:%S %p"]
                time_parsed = None
                
                for fmt in time_formats:
                    try:
                        time_parsed = datetime.strptime(time_str, fmt)
                        break
                    except ValueError:
                        continue
                        
                if time_parsed:
                    parsed_date = parsed_date.replace(
                        hour=time_parsed.hour,
                        minute=time_parsed.minute,
                        second=time_parsed.second
                    )
            
            # Format as ISO
            return parsed_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return ""

class TextMetrics:
    """Text similarity metrics for evaluation"""
    
    def __init__(self):
        self.rouge = None
        self.smooth = None
        
        if METRICS_AVAILABLE:
            self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
            self.smooth = SmoothingFunction().method1
    
    def rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L F1 score"""
        if not self.rouge or not reference or not candidate:
            return 0.0
        
        # Ensure strings and convert to lowercase
        reference = str(reference).lower().strip() if reference else ""
        candidate = str(candidate).lower().strip() if candidate else ""
        
        if not reference or not candidate:
            return 0.0
            
        scores = self.rouge.score(reference, candidate)
        return scores['rougeL'].fmeasure
        
    def bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score"""
        if not self.smooth or not reference or not candidate:
            return 0.0
            
        # Ensure strings and convert to lowercase
        reference = str(reference).lower().strip() if reference else ""
        candidate = str(candidate).lower().strip() if candidate else ""
        
        if not reference or not candidate:
            return 0.0
            
        ref_tokens = reference.split()
        cand_tokens = candidate.split()
        
        # Handle edge cases
        if not ref_tokens or not cand_tokens:
            return 0.0
            
        # BLEU needs a list of references
        return sentence_bleu([ref_tokens], cand_tokens, smoothing_function=self.smooth)
        
    def evaluate_text(self, reference: str, candidate: str) -> Dict[str, float]:
        """Evaluate text using multiple metrics"""
        # Ensure strings and normalize to lowercase
        reference = str(reference).lower().strip() if reference else ""
        candidate = str(candidate).lower().strip() if candidate else ""
        
        # Calculate exact match
        exact_match = 1.0 if reference == candidate else 0.0
        
        # Return metrics
        metrics = {
            'exact_match': exact_match,
            'rouge_l': self.rouge_l(reference, candidate),
            'bleu': self.bleu(reference, candidate)
        }
        
        # Calculate combined score
        if METRICS_AVAILABLE:
            metrics['combined_score'] = (metrics['rouge_l'] * 0.7 + metrics['bleu'] * 0.3)
        else:
            metrics['combined_score'] = exact_match
            
        return metrics

class CalendarEventExtractor:
    """Extract calendar events from model outputs"""
    
    def extract_from_json(self, text: str) -> Dict[str, Any]:
        """Extract a calendar event from JSON output"""
        if not text:
            return {}
            
        # Try to find a JSON object in the text
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, text)
        
        if match:
            try:
                json_text = match.group(1)
                result = json.loads(json_text)
                
                # Map 'type' to 'intent' if 'type' exists but 'intent' doesn't
                if 'type' in result and 'intent' not in result:
                    result['intent'] = result.pop('type')
                
                return result
            except json.JSONDecodeError:
                # If it's not valid JSON, fall back to regex extraction
                pass
                
        # If no valid JSON found or extraction failed, use regex as fallback
        return self.extract_from_text(text)
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract a calendar event from unstructured text using regex"""
        if not text:
            return {}
            
        result = {}
        
        # Define regex patterns for each field
        patterns = {
            'title': r'(?:title|event)[:\s]+([^\n,]+)',
            'intent': r'(?:intent|type)[:\s]+([^\n,]+)',  # Support both intent and type fields
            'description': r'(?:description)[:\s]+([^\n,]+)',
            'date': r'(?:date)[:\s]+([0-9-]+)',
            'startTime': r'(?:startTime|start time|start)[:\s]+([^\n,]+)',
            'endTime': r'(?:endTime|end time|end)[:\s]+([^\n,]+)',
            'location': r'(?:location)[:\s]+([^\n,]+)',
            'isAllDay': r'(?:isAllDay|all day)[:\s]+([^\n,]+)',
            'response': r'(?:response)[:\s]+([^\n]+)'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[field] = match.group(1).strip()
        
        # Map 'type' to 'intent' if 'type' exists but 'intent' doesn't
        if 'type' in result and 'intent' not in result:
            result['intent'] = result.pop('type')
                
        return result

class CalendarEventEvaluator:
    """Evaluator for calendar event extraction"""
    
    def __init__(self):
        self.text_metrics = TextMetrics()
        self.dt_utils = DateTimeUtils()
        
        # Define field types for specialized evaluation
        self.field_types = {
            'text': ['title', 'description', 'response'],
            'date': ['date'],
            'datetime': ['startTime', 'endTime'],
            'location': ['location'],
            'type': ['intent']
        }
        
        # Define weights for overall score calculation
        self.field_weights = {
            'title': 0.10,
            'description': 0.10,
            'date': 0.35,
            'startTime': 0.30,
            'endTime': 0.30,
            'location': 0.30,
            'intent': 0.50,
            'response': 0.10
        }
        
    def evaluate_text_field(self, reference: str, prediction: str) -> Dict[str, float]:
        """Evaluate a text field using Rouge-L and BLEU"""
        # Force lowercase for all text evaluations
        if reference:
            reference = str(reference).lower().strip()
        if prediction:
            prediction = str(prediction).lower().strip()
        return self.text_metrics.evaluate_text(reference, prediction)
        
    def evaluate_date_field(self, reference: str, prediction: str) -> Dict[str, float]:
        """Evaluate a date field"""
        valid_format = self.dt_utils.is_valid_date_format(prediction) if prediction else False
        
        # For comparing dates, normalize both to YYYY-MM-DD if needed
        ref_date = reference
        pred_date = prediction
        
        # Try to extract date from ISO datetime if needed
        if reference:
            if self.dt_utils.is_valid_date_format(reference):
                ref_date = reference
            elif self.dt_utils.is_valid_iso_datetime(reference):
                ref_date = self.dt_utils.extract_date_from_iso(reference)
            else:
                # Try to convert non-standard format to ISO
                iso_date = self.dt_utils.convert_to_iso_format(reference)
                if iso_date:
                    ref_date = self.dt_utils.extract_date_from_iso(iso_date)
            
        if prediction and not self.dt_utils.is_valid_date_format(prediction) and self.dt_utils.is_valid_iso_datetime(prediction):
            pred_date = self.dt_utils.extract_date_from_iso(prediction)
        
        # Check for exact match of the date values
        value_match = False
        if ref_date and pred_date:
            try:
                ref_dt = datetime.strptime(ref_date, "%Y-%m-%d")
                pred_dt = datetime.strptime(pred_date, "%Y-%m-%d")
                value_match = ref_dt == pred_dt
            except ValueError:
                value_match = False
                
        return {
            'format_valid': 1.0 if valid_format else 0.0,
            'value_match': 1.0 if value_match else 0.0,
            'overall': 1.0 if (valid_format and value_match) else 0.0
        }
        
    def evaluate_datetime_field(self, reference: str, prediction: str) -> Dict[str, float]:
        """Evaluate a datetime field"""
        valid_format = self.dt_utils.is_valid_iso_datetime(prediction) if prediction else False
        
        # Normalize reference to ISO format if needed
        ref_iso = reference
        if reference and not self.dt_utils.is_valid_iso_datetime(reference):
            # Try to convert date and time to ISO format
            ref_date = None
            ref_time = None
            
            # If reference contains both date and time, try to extract them
            if ' ' in reference:
                parts = reference.split(' ')
                for part in parts:
                    if self.dt_utils.is_valid_date_format(part):
                        ref_date = part
                    elif ':' in part:
                        ref_time = part
            
            # If we found potential date/time, convert to ISO
            if ref_date or ref_time:
                ref_iso = self.dt_utils.convert_to_iso_format(ref_date or reference, ref_time)
        
        # Compare the datetime values
        value_match = self.dt_utils.compare_iso_datetimes(ref_iso, prediction) if (ref_iso and prediction) else False
        
        # Check time part separately
        ref_time = self.dt_utils.extract_time_from_iso(ref_iso) if ref_iso else ""
        pred_time = self.dt_utils.extract_time_from_iso(prediction) if prediction else ""
        time_match = ref_time == pred_time
        
        return {
            'format_valid': 1.0 if valid_format else 0.0,
            'value_match': 1.0 if value_match else 0.0,
            'time_match': 1.0 if time_match else 0.0,
            'overall': 1.0 if (valid_format and value_match) else 0.0
        }
        
    def evaluate_category_field(self, reference: str, prediction: str) -> Dict[str, float]:
        """Evaluate a categorical field (e.g., event type)"""
        if reference is None:
            reference = ""
        if prediction is None:
            prediction = ""
            
        reference = str(reference).lower().strip()
        prediction = str(prediction).lower().strip()
        
        return {
            'match': 1.0 if reference == prediction else 0.0
        }
        
    def evaluate_location_field(self, reference: str, prediction: str) -> Dict[str, float]:
        """Evaluate a location field with exact, partial and semantic matching"""
        if reference is None:
            reference = ""
        if prediction is None:
            prediction = ""
            
        reference = str(reference).lower().strip()
        prediction = str(prediction).lower().strip()
        
        # Exact match check
        exact_match = 1.0 if reference == prediction else 0.0
        
        # Partial match (if one is contained in the other)
        partial_match = 0.0
        if reference and prediction:
            if reference in prediction or prediction in reference:
                partial_match = 0.7  # Partial match score
        
        # Word overlap score (count common words)
        word_overlap = 0.0
        if reference and prediction:
            ref_words = set(reference.split())
            pred_words = set(prediction.split())
            
            if ref_words and pred_words:
                common_words = ref_words.intersection(pred_words)
                word_overlap = len(common_words) / max(len(ref_words), len(pred_words))
        
        # Calculate overall score (prioritize exact match, then partial, then word overlap)
        overall = max(exact_match, partial_match, word_overlap)
        
        return {
            'exact_match': exact_match,
            'partial_match': partial_match,
            'word_overlap': word_overlap,
            'overall': overall
        }
        
    def evaluate_field(self, field_name: str, reference: Any, prediction: Any) -> Dict[str, float]:
        """Evaluate a single field based on its type"""
        # Handle None values
        if reference is None and prediction is None:
            return {'skipped': True}
        
        # Determine field type and use appropriate evaluation
        if field_name in self.field_types['text']:
            return self.evaluate_text_field(reference, prediction)
        elif field_name in self.field_types['date']:
            return self.evaluate_date_field(reference, prediction)
        elif field_name in self.field_types['datetime']:
            return self.evaluate_datetime_field(reference, prediction)
        elif field_name in self.field_types['location']:
            return self.evaluate_location_field(reference, prediction)
        elif field_name in self.field_types['type']:
            return self.evaluate_category_field(reference, prediction)
        else:
            # Default to text evaluation if field type isn't specified
            return self.evaluate_text_field(reference, prediction)
    
    def compute_type_confusion_matrix(self, references: List[str], predictions: List[str]) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        """Compute confusion matrix and metrics for event type classification"""
        # Get unique class labels
        unique_labels = sorted(list(set([r for r in references if r] + [p for p in predictions if p])))
        
        # Create label mappings
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        
        # Convert to numeric labels for confusion matrix
        y_true = [label_to_id.get(r, -1) if r else -1 for r in references]
        y_pred = [label_to_id.get(p, -1) if p else -1 for p in predictions]
        
        # Filter out missing values (-1)
        valid_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != -1 and p != -1]
        valid_true = [y_true[i] for i in valid_indices]
        valid_pred = [y_pred[i] for i in valid_indices]
        
        # If not enough data, return empty results
        if len(valid_true) < 2:
            return np.array([]), {}
        
        # Compute confusion matrix
        cm = confusion_matrix(valid_true, valid_pred, labels=range(len(unique_labels)))
        
        # Calculate metrics
        accuracy = accuracy_score(valid_true, valid_pred)
        f1 = f1_score(valid_true, valid_pred, average='weighted')
        
        metrics = {
            "accuracy": accuracy,
            "f1_score": f1,
            "class_names": unique_labels,
            "true_labels": valid_true,
            "pred_labels": valid_pred
        }
        
        return cm, metrics
        
    def evaluate_event(self, reference: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a complete calendar event"""
        if not reference:
            # Cannot evaluate without a reference
            return {'error': 'No reference event provided'}
            
        if not prediction:
            # Empty prediction
            return {
                'fields': {field: {'error': 'Missing field'} for field in reference},
                'completeness': 0.0,
                'overall_score': 0.0
            }
            
        # Track field completeness
        total_fields = 0
        available_fields = 0
        
        # Evaluate each field
        field_scores = {}
        overall_scores = []
        
        # Handle type->intent field mapping in prediction if needed
        if 'type' in prediction and 'intent' not in prediction:
            prediction['intent'] = prediction.pop('type')
            
        for field in set(reference.keys()) | set(prediction.keys()):
            # Skip internal fields and isAllDay (as specified)
            if field.startswith('_') or field == 'isAllDay':
                continue
                
            ref_value = reference.get(field)
            pred_value = prediction.get(field)
            
            total_fields += 1
            if field in prediction:
                available_fields += 1
                
            # Evaluate this field
            field_result = self.evaluate_field(field, ref_value, pred_value)
            
            # Get a single score value for this field
            field_score = 0.0
            if 'skipped' in field_result and field_result['skipped']:
                field_score = None  # Skip this field
            elif 'overall' in field_result:
                field_score = field_result['overall']
            elif 'combined_score' in field_result:
                field_score = field_result['combined_score']
            elif 'match' in field_result:
                field_score = field_result['match']
            elif 'exact_match' in field_result:
                field_score = field_result['exact_match']
            
            # Store result
            field_scores[field] = field_result
            
            # Add to overall calculation if we have a weight for this field
            if field_score is not None and field in self.field_weights:
                overall_scores.append((field, field_score * self.field_weights[field]))
                
        # Calculate completeness
        completeness = available_fields / total_fields if total_fields > 0 else 0.0
        
        # Calculate overall score
        total_weight = sum(self.field_weights[field] for field, _ in overall_scores)
        overall_score = sum(score for _, score in overall_scores) / total_weight if total_weight > 0 else 0.0
        
        return {
            'fields': field_scores,
            'completeness': completeness,
            'overall_score': overall_score
        }
        
    def evaluate_batch(self, references: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of events and compute aggregate metrics"""
        if len(references) != len(predictions):
            raise ValueError(f"Number of references ({len(references)}) does not match number of predictions ({len(predictions)})")
            
        # Evaluate each event
        event_results = []
        for ref, pred in zip(references, predictions):
            result = self.evaluate_event(ref, pred)
            event_results.append(result)
            
        # Collect scores for each field
        field_scores = defaultdict(list)
        completeness_scores = []
        overall_scores = []
        
        # Also collect intent values for confusion matrix
        intent_references = []
        intent_predictions = []
        
        for ref, pred, result in zip(references, predictions, event_results):
            if 'error' in result:
                continue
                
            completeness_scores.append(result['completeness'])
            overall_scores.append(result['overall_score'])
            
            # Collect intent values
            if 'intent' in ref and 'intent' in pred:
                intent_references.append(str(ref['intent']).lower().strip())
                intent_predictions.append(str(pred['intent']).lower().strip())
                
            for field, field_result in result['fields'].items():
                # Skip internal fields and errors
                if field.startswith('_') or 'error' in field_result:
                    continue
                    
                # Extract all scores for this field
                if field in self.field_types['text']:
                    for metric_name, metric_value in field_result.items():
                        field_scores[f"{field}_{metric_name}"].append(metric_value)
                
                # Extract the main score for this field
                if 'overall' in field_result:
                    field_scores[field].append(field_result['overall'])
                elif 'combined_score' in field_result:
                    field_scores[field].append(field_result['combined_score'])
                elif 'match' in field_result:
                    field_scores[field].append(field_result['match'])
                elif 'exact_match' in field_result:
                    field_scores[field].append(field_result['exact_match'])
                    
        # Calculate aggregate statistics
        aggregate_results = {
            'num_events': len(references),
            'completeness': {
                'mean': float(np.mean(completeness_scores)) if completeness_scores else 0.0,
                'std': float(np.std(completeness_scores)) if completeness_scores else 0.0
            },
            'overall_score': {
                'mean': float(np.mean(overall_scores)) if overall_scores else 0.0,
                'std': float(np.std(overall_scores)) if overall_scores else 0.0
            },
            'field_scores': {}
        }
        
        # Calculate stats for each field
        for field, scores in field_scores.items():
            aggregate_results['field_scores'][field] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores)
            }
            
        # Add confusion matrix for intent field if we have data
        if intent_references and intent_predictions:
            cm, metrics = self.compute_type_confusion_matrix(intent_references, intent_predictions)
            if len(cm) > 0:
                aggregate_results['intent_confusion_matrix'] = metrics
                # Store the raw confusion matrix as well
                aggregate_results['intent_confusion_matrix']['matrix'] = cm.tolist()
                
                # Use intent accuracy as the intent field score
                aggregate_results['field_scores']['intent'] = {
                    'mean': float(metrics['accuracy']),
                    'std': 0.0,  # No std dev for a single accuracy value
                    'count': len(intent_references)
                }
            
        # Add detailed results
        aggregate_results['event_results'] = event_results
        
        return aggregate_results

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a file (JSON, JSONL, or CSV)"""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df.to_dict('records')
        
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
        
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def get_model_predictions(model, tokenizer, prompts: List[str], system_prompt: str = None, 
                         max_length: int = 512, batch_size: int = 8) -> List[str]:
    """Generate predictions from the model for a list of prompts"""
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating predictions"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Format inputs with system prompt if provided
        if system_prompt:
            formatted_prompts = [f"{system_prompt}\n\nUser: {prompt}" for prompt in batch_prompts]
        else:
            formatted_prompts = batch_prompts
        
        inputs = tokenizer(formatted_prompts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=False
            )
            
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        for prompt, prediction in zip(formatted_prompts, batch_predictions):
            if prediction.startswith(prompt):
                prediction = prediction[len(prompt):].strip()
            predictions.append(prediction)
    
    return predictions

def process_model_outputs(raw_outputs: List[str]) -> List[Dict[str, Any]]:
    """Process raw model outputs into structured event data"""
    extractor = CalendarEventExtractor()
    processed_outputs = []
    
    for output in raw_outputs:
        event = extractor.extract_from_json(output)
        processed_outputs.append(event)
        
    return processed_outputs
