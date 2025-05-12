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
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import dateutil.parser
from collections import defaultdict
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import torch
from tqdm import tqdm
from openai import OpenAI


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
                try:
                    # Check if it's a whole number and convert to int first if so
                    if date_str.is_integer():
                        date_str = str(int(date_str))
                    else:
                        date_str = str(date_str)
                except AttributeError:
                    date_str = str(date_str)
            elif isinstance(date_str, int):
                date_str = str(date_str)
            elif not isinstance(date_str, str):
                date_str = str(date_str)
                
            # Try to parse the date
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except (ValueError, TypeError, AttributeError):
            return False
            
    @staticmethod
    def is_valid_iso_datetime(dt_str: str) -> bool:
        """Check if string is a valid ISO datetime format"""
        if not dt_str:
            return False
        # Handle float values
        if isinstance(dt_str, float):
            try:
                dt_str = str(dt_str)
            except:
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
        # Handle float values
        if isinstance(dt_str, float):
            try:
                dt_str = str(dt_str)
            except:
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
        # Handle float values
        if isinstance(dt_str, float):
            try:
                dt_str = str(dt_str)
            except:
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
            
        # Handle float values
        if isinstance(dt1, float):
            try:
                dt1 = str(dt1)
            except:
                return False
                
        if isinstance(dt2, float):
            try:
                dt2 = str(dt2)
            except:
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
            print("DEBUG: Empty text provided to extract_from_json")
            return {}
            
        # Try to find a JSON object in the text
        json_pattern = r'(\{[\s\S]*\})'
        match = re.search(json_pattern, text)
        
        if match:
            try:
                json_text = match.group(1)
                print(f"DEBUG: Found JSON text: {json_text[:100]}...")
                result = json.loads(json_text)
                
                # Map 'type' to 'intent' if 'type' exists but 'intent' doesn't
                if 'type' in result and 'intent' not in result:
                    result['intent'] = result.pop('type')
                
                return result
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON decode error: {e}")
                # If it's not valid JSON, fall back to regex extraction
                pass
        else:
            print(f"DEBUG: No JSON pattern found in text: {text[:100]}...")
                
        # If no valid JSON found or extraction failed, use regex as fallback
        return self.extract_from_text(text)
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract a calendar event from unstructured text using regex"""
        if not text:
            print("DEBUG: Empty text provided to extract_from_text")
            return {}
            
        result = {}
        
        # First, try to extract JSON-like structures with key-value pairs
        json_like_pattern = r'"([^"]+)"\s*:\s*"([^"]*)"'
        json_like_matches = re.findall(json_like_pattern, text)
        
        if json_like_matches:
            print(f"DEBUG: Found {len(json_like_matches)} JSON-like key-value pairs")
            for key, value in json_like_matches:
                if key in ["title", "intent", "description", "date", "startTime", "endTime", "location", "response"]:
                    result[key] = value
            
            # Handle isAllDay separately as it's a number, not a string
            isAllDay_match = re.search(r'"isAllDay"\s*:\s*(\d)', text)
            if isAllDay_match:
                result["isAllDay"] = int(isAllDay_match.group(1))
                
            # If we found at least some fields, return the result
            if result:
                print(f"DEBUG: Extracted {len(result)} fields from JSON-like structure")
                return result
        
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
        
        # Look for intent values directly in the text
        intent_values = ["add", "update", "delete", "query", "chitchat"]
        for intent in intent_values:
            if re.search(r'\b' + intent + r'\b', text, re.IGNORECASE):
                result["intent"] = intent
                break
        
        # Map 'type' to 'intent' if 'type' exists but 'intent' doesn't
        if 'type' in result and 'intent' not in result:
            result['intent'] = result.pop('type')
        
        print(f"DEBUG: Extracted {len(result)} fields using regex: {', '.join(result.keys())}")
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
            'date': 0.50,
            'startTime': 0.50,
            'endTime': 0.20,
            'location': 0.40,
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
        
        # Handle float reference
        if isinstance(reference, float):
            try:
                reference = str(reference)
            except:
                reference = ""
        
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
        
        # Handle float prediction
        if isinstance(prediction, float):
            try:
                prediction = str(prediction)
            except:
                prediction = ""
            
        if prediction and not self.dt_utils.is_valid_date_format(prediction) and self.dt_utils.is_valid_iso_datetime(prediction):
            pred_date = self.dt_utils.extract_date_from_iso(prediction)
        
        # Check for exact match of the date values
        value_match = False
        if ref_date and pred_date:
            try:
                # Make sure both are strings before parsing
                ref_date_str = str(ref_date)
                pred_date_str = str(pred_date)
                ref_dt = datetime.strptime(ref_date_str, "%Y-%m-%d")
                pred_dt = datetime.strptime(pred_date_str, "%Y-%m-%d")
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
        
        # Handle float values by converting to string
        if isinstance(reference, float):
            try:
                # Try to convert float to string representation of a datetime
                ref_iso = str(reference)
            except:
                ref_iso = ""
        
        if reference and not isinstance(reference, float) and not self.dt_utils.is_valid_iso_datetime(reference):
            # Try to convert date and time to ISO format
            ref_date = None
            ref_time = None
            
            # If reference contains both date and time, try to extract them
            if isinstance(reference, str) and ' ' in reference:
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
        if reference is None:
            reference = ""
        if prediction is None:
            prediction = ""
            
        # For empty strings or None values in both reference and prediction, consider it a match
        if (reference == "" or reference is None) and (prediction == "" or prediction is None):
            if field_name in self.field_types['text']:
                return {'exact_match': 1.0, 'rouge_l': 1.0, 'bleu': 1.0, 'combined_score': 1.0}
            elif field_name in self.field_types['location']:
                return {'exact_match': 1.0, 'partial_match': 1.0, 'word_overlap': 1.0, 'overall': 1.0}
            elif field_name in self.field_types['type']:
                return {'match': 1.0}
            else:
                return {'overall': 1.0}
        
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
        
        # Get reference intent (to determine which fields to evaluate)
        reference_intent = str(reference.get('intent', '')).lower().strip()
        
        # Define fields to evaluate based on intent type
        fields_to_evaluate = set(reference.keys()) | set(prediction.keys())
        
        # Remove 'text' field from fields to evaluate
        if 'text' in fields_to_evaluate:
            fields_to_evaluate.remove('text')
            
        # Add 'response' field if missing in reference but present in prediction
        if 'response' not in reference and 'response' in prediction:
            reference['response'] = ""  # Use empty string for missing response field
        
        # For CANCEL intent, only evaluate intent, date, startTime, and response
        if reference_intent == 'cancel':
            fields_to_evaluate = {'intent', 'date', 'startTime', 'response', 'title'}
            
        # For QUERY intent, only evaluate intent, date, startTime, and response
        elif reference_intent == 'query':
            fields_to_evaluate = {'intent', 'date', 'startTime', 'response', 'title'}
            
        # For CHITCHAT intent, only evaluate intent and response
        elif reference_intent == 'chitchat':
            fields_to_evaluate = {'intent', 'response'}
        
        # For ADD or UPDATE intent, evaluate all fields (default behavior)
        # Also default to all fields if intent is not recognized
            
        for field in fields_to_evaluate:
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
                
        # Calculate completeness (based on evaluated fields only)
        fields_in_prediction = set(prediction.keys()) & fields_to_evaluate
        fields_in_reference = set(reference.keys()) & fields_to_evaluate
        available_fields = len(fields_in_prediction)
        total_fields = len(fields_in_reference)
        completeness = available_fields / total_fields if total_fields > 0 else 0.0
        
        # Calculate overall score
        total_weight = sum(self.field_weights[field] for field, _ in overall_scores)
        overall_score = sum(score for _, score in overall_scores) / total_weight if total_weight > 0 else 0.0
        
        return {
            'fields': field_scores,
            'completeness': completeness,
            'overall_score': overall_score,
            'reference_intent': reference_intent  # Add reference intent for clarity
        }
        
    def evaluate_batch(self, references: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of events and compute aggregate metrics"""
        if len(references) != len(predictions):
            raise ValueError(f"Number of references ({len(references)}) does not match number of predictions ({len(predictions)})")
            
        # Process references to extract actual event data
        processed_references = []
        for ref in references:
            # Check if the reference contains a text field with JSON data
            if 'text' in ref and '<|assistant|>' in ref['text']:
                # Extract the JSON part after <|assistant|> tag
                assistant_part = ref['text'].split('<|assistant|>')[1].strip()
                # Remove <|end|> tag if present
                if '<|end|>' in assistant_part:
                    assistant_part = assistant_part.split('<|end|>')[0].strip()
                
                # Try to parse the JSON
                try:
                    event_data = json.loads(assistant_part)
                    processed_references.append(event_data)
                    continue
                except json.JSONDecodeError:
                    pass
                    
            # If we couldn't extract JSON or there's no text field, use the reference as is
            processed_references.append(ref)
            
        # Print the first few references and predictions for debugging
        print("\n===== REFERENCE VS PREDICTION COMPARISON =====")
        for i in range(min(3, len(processed_references))):
            print(f"\nSample {i+1}:")
            print(f"Reference: {json.dumps(processed_references[i], indent=2)}")
            print(f"Prediction: {json.dumps(predictions[i], indent=2)}")
            
        # Evaluate each event
        event_results = []
        for ref, pred in zip(processed_references, predictions):
            result = self.evaluate_event(ref, pred)
            event_results.append(result)
            
        # Collect scores for each field
        field_scores = defaultdict(list)
        completeness_scores = []
        overall_scores = []
        
        # Also collect intent values for confusion matrix
        intent_references = []
        intent_predictions = []
        
        for ref, pred, result in zip(processed_references, predictions, event_results):
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
            'num_events': len(processed_references),
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
            data = json.load(f)
    
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        data = df.to_dict('records')
        
    elif file_path.endswith('.jsonl'):
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Process data to extract user text and reference JSON
    processed_data = []
    for item in data:
        if 'text' in item:
            # Extract user text from the ChatML format
            text = item['text']
            
            # Extract the user query part
            user_text = ""
            if "<|user|>" in text:
                user_parts = text.split("<|user|>")
                if len(user_parts) > 1:
                    user_text = user_parts[1].split("<|assistant|>")[0].strip()
            
            # If we couldn't extract user text, use the original text
            if not user_text:
                user_text = text
                
            # Create a new item with just the user text
            new_item = {'text': user_text}
            
            # Extract reference JSON from assistant part
            if "<|assistant|>" in text:
                assistant_part = text.split("<|assistant|>")[1].strip()
                if "<|end|>" in assistant_part:
                    assistant_part = assistant_part.split("<|end|>")[0].strip()
                
                # Try to parse the JSON
                try:
                    reference_json = json.loads(assistant_part)
                    # Add reference fields to the new item
                    for key, value in reference_json.items():
                        new_item[key] = value
                except json.JSONDecodeError:
                    pass
            
            processed_data.append(new_item)
        else:
            processed_data.append(item)
    
    # Print sample of processed data
    if processed_data:
        print("\n===== PROCESSED TEST DATA SAMPLE =====")
        for i in range(min(3, len(processed_data))):
            print(f"\nSample {i+1}:")
            print(f"User text: {processed_data[i].get('text', '')[:100]}...")
            for key, value in processed_data[i].items():
                if key != 'text':
                    print(f"{key}: {value}")
    
    return processed_data

def get_model_predictions(model, tokenizer, prompts: List[str], system_prompt: str = None, 
                         max_new_tokens: int = 256, batch_size: int = 8) -> List[str]:
    """Generate predictions from the model for a list of prompts"""
    predictions = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating predictions"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Format inputs with ChatML format
        formatted_prompts = []
        for prompt in batch_prompts:
            # Remove any existing system or assistant tags from the user prompt
            clean_prompt = prompt
            if "<|system|>" in clean_prompt:
                clean_prompt = clean_prompt.split("<|system|>")[-1]
            if "<|user|>" in clean_prompt:
                clean_prompt = clean_prompt.split("<|user|>")[-1]
            if "<|assistant|>" in clean_prompt:
                clean_prompt = clean_prompt.split("<|assistant|>")[0]
            
            # Clean up any remaining tags
            clean_prompt = re.sub(r'<\|.*?\|>', '', clean_prompt).strip()
            
            # Format with proper ChatML tags
            if system_prompt:
                formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{clean_prompt}\n<|assistant|>"
            else:
                formatted_prompt = f"<|user|>\n{clean_prompt}\n<|assistant|>"
            
            formatted_prompts.append(formatted_prompt)
        
        # Print input length for debugging
        if i == 0:
            sample_tokens = tokenizer(formatted_prompts[0], return_tensors="pt")
            print(f"\nDEBUG: Input length of first prompt: {len(sample_tokens['input_ids'][0])} tokens")
        
        inputs = tokenizer(formatted_prompts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False
            )
            
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Print debug info for first batch
        if i == 0:
            print("\n===== DEBUGGING PROMPT AND RESPONSE EXTRACTION =====")
            for j, (prompt, full_output) in enumerate(zip(formatted_prompts[:1], batch_predictions[:1])):
                print(f"\nSample {j+1}:")
                print(f"Formatted prompt: {repr(prompt)}")
                print(f"Full model output: {repr(full_output)}")
        
        # Extract only the assistant's response part
        for prompt, prediction in zip(formatted_prompts, batch_predictions):
            # Extract JSON objects directly
            json_pattern = r'\{"title":".*?","intent":".*?","description":".*?","date":".*?","startTime":".*?","endTime":".*?","location":".*?","isAllDay":[01].*?\}'
            matches = re.findall(json_pattern, prediction)
            
            if matches:
                # Use the first JSON object found
                predictions.append(matches[0])
                continue
                
            # If no JSON object found, try to extract the assistant's response
            if prediction.startswith(prompt):
                # Extract content after <|assistant|> tag
                assistant_part = prediction[len(prompt):].strip()
                # Remove <|end|> tag if present
                if "<|end|>" in assistant_part:
                    assistant_part = assistant_part.split("<|end|>")[0].strip()
                predictions.append(assistant_part)
            else:
                # Fallback: try to find the assistant's response using regex
                match = re.search(r"<\|assistant\|>(.*?)(?:<\|end\|>|$)", prediction, re.DOTALL)
                if match:
                    predictions.append(match.group(1).strip())
                else:
                    # If all else fails, just return the raw output
                    predictions.append(prediction)
    
    return predictions

def get_gpt_predictions(prompts, system_prompt=None, model="gpt-4o-mini"):
    """Generate predictions using GPT for a list of prompts"""
    # Initialize the client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    predictions = []
    
    for prompt in prompts:
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Get response from GPT using the new API syntax
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,  # Use 0 for more deterministic outputs
            max_tokens=512
        )
        
        # Extract the response text
        prediction = response.choices[0].message.content
        predictions.append(prediction)
        
    return predictions

def process_model_outputs(raw_outputs: List[str]) -> List[Dict[str, Any]]:
    """Process raw model outputs into structured event data"""
    processed_outputs = []
    
    print("\n===== RAW MODEL OUTPUTS =====")
    for i, output in enumerate(raw_outputs):
        print(f"\n--- Prediction {i+1} ---")
        print(f"Raw output: {repr(output)}")
        
        # Clean up the output - remove any ChatML tags that might remain
        output = re.sub(r'<\|.*?\|>', '', output).strip()
        
        # Extract JSON objects using regex - look for complete JSON objects
        json_pattern = r'\{"title":".*?","intent":".*?","description":".*?","date":".*?","startTime":".*?","endTime":".*?","location":".*?","isAllDay":[01](?:,"response":".*?")?\}'
        matches = re.findall(json_pattern, output)
        
        event = {}
        if matches:
            # Use the first complete JSON object found
            try:
                print(f"Found JSON: {matches[0]}")
                event = json.loads(matches[0])
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                # If that fails, fall back to regex extraction
                event = {}
        
        # If no valid JSON found, try a more aggressive approach with the extractor
        if not event:
            extractor = CalendarEventExtractor()
            event = extractor.extract_from_text(output)
            
        print(f"Extracted event: {json.dumps(event, ensure_ascii=False)}")
        processed_outputs.append(event)
    
    print("\n===== END OF RAW OUTPUTS =====\n")
    return processed_outputs

def setup_pretrained_model_and_tokenizer(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_path="ShenghaoYummy/calendar-assistant_v4",
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # 1. Load the tokenizer that *matches the adapter* (adapter repo always has one)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # 2. Load the base model *without forcing a new vocab size*
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )

    # 3. Resize to **exactly** the adapter’s vocab length *before* attaching LoRA
    base_model.resize_token_embeddings(len(tokenizer))   # 32 000 → 32 000, no change

    # 4. Attach LoRA deltas
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer

def print_detailed_results(references, predictions, results):
    """Print detailed results for each prediction"""
    print("\nDetailed Results:")
    print("=" * 80)
    
    # Process references to extract actual event data
    processed_references = []
    for ref in references:
        # Check if the reference contains a text field with JSON data
        if 'text' in ref and '<|assistant|>' in ref['text']:
            # Extract the JSON part after <|assistant|> tag
            assistant_part = ref['text'].split('<|assistant|>')[1].strip()
            # Remove <|end|> tag if present
            if '<|end|>' in assistant_part:
                assistant_part = assistant_part.split('<|end|>')[0].strip()
            
            # Try to parse the JSON
            try:
                event_data = json.loads(assistant_part)
                processed_references.append(event_data)
                continue
            except json.JSONDecodeError:
                pass
                
        # If we couldn't extract JSON or there's no text field, use the reference as is
        processed_references.append(ref)
    
    for i, (ref, pred, event_result) in enumerate(zip(processed_references, predictions, results['event_results'])):
        print(f"\nSample {i+1}:")
        print("-" * 40)
        
        # Print reference intent (if available)
        if 'reference_intent' in event_result:
            print(f"Reference Intent: {event_result['reference_intent'].upper()}")
            print("Fields evaluated based on this intent type")
            print("-" * 40)
        
        # Print reference
        print("Reference:")
        for field, value in ref.items():
            if field not in ['_id', '_source', 'text']:  # Skip internal fields and text field
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
    
    # Count samples by intent type
    intent_counts = {}
    for event_result in results['event_results']:
        if 'reference_intent' in event_result:
            intent = event_result['reference_intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    # Print intent type statistics
    if intent_counts:
        print("\nSamples by Intent Type:")
        for intent, count in sorted(intent_counts.items()):
            print(f"  {intent.upper()}: {count} samples")
    
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