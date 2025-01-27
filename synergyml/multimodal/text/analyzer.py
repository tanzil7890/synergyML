"""Text analysis module for SynergyML."""

from typing import Dict, Any, Optional
from transformers import pipeline
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter

class TextAnalyzer:
    """Text analysis class for processing transcripts and textual content."""
    
    def __init__(
        self,
        language: str = "en",
        use_gpu: bool = False
    ):
        """Initialize text analyzer.
        
        Parameters
        ----------
        language : str
            Language code (default: "en")
        use_gpu : bool
            Whether to use GPU acceleration
        """
        # Initialize NLP pipelines
        self.nlp = spacy.load(f"{language}_core_web_sm")
        
        # Download required NLTK data
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformers pipeline for zero-shot classification
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if use_gpu else -1
        )
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        Dict[str, float]
            Sentiment scores
        """
        return self.sentiment_analyzer.polarity_scores(text)
    
    def extract_key_phrases(self, text: str, top_k: int = 10) -> list:
        """Extract key phrases from text.
        
        Parameters
        ----------
        text : str
            Input text
        top_k : int
            Number of key phrases to extract
            
        Returns
        -------
        list
            List of key phrases
        """
        doc = self.nlp(text)
        phrases = []
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit phrase length
                phrases.append(chunk.text)
        
        # Count frequencies
        phrase_freq = Counter(phrases)
        
        return [phrase for phrase, _ in phrase_freq.most_common(top_k)]
    
    def classify_content(
        self,
        text: str,
        categories: list
    ) -> Dict[str, float]:
        """Classify text into categories.
        
        Parameters
        ----------
        text : str
            Input text
        categories : list
            List of categories for classification
            
        Returns
        -------
        Dict[str, float]
            Classification scores
        """
        results = self.classifier(text, categories)
        return dict(zip(results['labels'], results['scores']))
    
    def extract_entities(self, text: str) -> Dict[str, list]:
        """Extract named entities from text.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        Dict[str, list]
            Dictionary of entity types and their mentions
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    def analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze text structure.
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        Dict[str, Any]
            Structural analysis results
        """
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text)
        
        # Analyze each sentence
        sentence_info = []
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            pos_tags = nltk.pos_tag(tokens)
            
            sentence_info.append({
                'text': sent,
                'length': len(tokens),
                'pos_distribution': Counter(tag for _, tag in pos_tags)
            })
        
        return {
            'sentence_count': len(sentences),
            'sentences': sentence_info,
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences)
        }
    
    def analyze(
        self,
        text: str,
        categories: Optional[list] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive text analysis.
        
        Parameters
        ----------
        text : str
            Input text
        categories : Optional[list]
            Categories for classification
            
        Returns
        -------
        Dict[str, Any]
            Combined analysis results
        """
        results = {
            'sentiment': self.analyze_sentiment(text),
            'key_phrases': self.extract_key_phrases(text),
            'entities': self.extract_entities(text),
            'structure': self.analyze_structure(text)
        }
        
        if categories:
            results['classification'] = self.classify_content(text, categories)
        
        return results 