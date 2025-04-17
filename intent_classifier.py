import pandas as pd
import numpy as np
import pickle
import re
import os

class IntentClassifier:
    def __init__(self):
        self.model = None
        self.intents = ['GREETING', 'FINANCE_QUESTION', 'DOCUMENT_QUESTION', 'OFF_TOPIC', 'HELP']
        
    def preprocess_text(self, text):
        """Clean and normalize text input"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text.strip()
    
    def predict_intent(self, query):
        """Predict the intent of a user query"""
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Handle empty queries
        if not processed_query:
            return 'OFF_TOPIC', 1.0
        
        # Make prediction
        intent = self.model.predict([processed_query])[0]
        
        # Get confidence scores for all classes
        proba = self.model.predict_proba([processed_query])[0]
        confidence = max(proba)
        
        return intent, confidence
    
    def load_model(self, filepath='kuberai_intent_classifier.pkl'):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Intent classifier model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def get_rule_based_intent(self, query):
        """Use rule-based patterns to determine intent for high-confidence cases"""
        query = query.lower().strip()
        
        # Expanded finance keywords
        finance_keywords = [
            'interest', 'compound', 'simple', 'rate', 'apr', 'apy', 'stock', 'bond', 
            'mutual fund', 'etf', 'index', 'dividend', 'yield', 'market', 'equity', 
            'debt', 'loan', 'mortgage', 'investment', 'portfolio', 'diversification',
            'asset', 'liability', 'balance sheet', 'income', 'expense', 'profit',
            'loss', 'revenue', 'cash flow', 'depreciation', 'amortization',
            'capital', 'liquidity', 'solvency', 'risk', 'return', 'roi', 'irr'
        ]
        
        # Greetings detection (very short and common greetings)
        if query in ['hi', 'hello', 'hey', 'yo', 'hi there', 'hello there'] or len(query.split()) <= 2 and any(g in query for g in ['hi', 'hello', 'hey', 'morning', 'afternoon']):
            return 'GREETING', 1.0
        
        # Document questions (explicit mentions)
        doc_keywords = ['document', 'pdf', 'report', 'upload', 'file', 'table', 'extracted']
        if any(keyword in query for keyword in doc_keywords):
            return 'DOCUMENT_QUESTION', 1.0
        
        # Help requests (explicit mentions)
        help_keywords = ['help', 'how do i use', 'what can you do', 'tutorial', 'guide']
        if any(keyword in query for keyword in help_keywords):
            return 'HELP', 1.0
        
        # Off-topic detection for common non-finance topics
        off_topic_keywords = ['weather', 'joke', 'recipe', 'movie', 'song', 'sport', 'game', 'news', 'politics']
        if any(keyword in query for keyword in off_topic_keywords):
            return 'OFF_TOPIC', 1.0
        
        # Finance questions (specific terms)
        if any(keyword in query for keyword in finance_keywords):
            # Count how many finance keywords appear for weighting
            finance_count = sum(1 for keyword in finance_keywords if keyword in query)
            if finance_count >= 1:
                return 'FINANCE_QUESTION', 1.0
            
        # If no rule matches, return None and let the ML model decide
        return None, 0.0
        
    def classify(self, query):
        """Combine rule-based and ML approaches for intent classification"""
        # First try rule-based classification for high-confidence cases
        rule_intent, rule_confidence = self.get_rule_based_intent(query)
        if rule_intent:
            return rule_intent, rule_confidence
            
        # Fall back to ML model
        if self.model:
            return self.predict_intent(query)
        else:
            # If model isn't loaded yet, use a simple fallback
            if len(query.split()) <= 3:
                return 'GREETING', 0.7
            else:
                return 'FINANCE_QUESTION', 0.5