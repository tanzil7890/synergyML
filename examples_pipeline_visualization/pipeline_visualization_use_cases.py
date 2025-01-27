"""Real-world use cases for pipeline visualization in SynergyML."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from synergyml.multimodal.llm_integration import LLMEnhancedAnalyzer
from synergyml.config import SynergyMLConfig

def analyze_content_generation_pipeline():
    """Analyze performance of content generation pipeline.
    
    Use Case: Content creation team monitoring LLM performance for
    different types of content (articles, social media, technical docs).
    """
    analyzer = LLMEnhancedAnalyzer(llm_backend='gpt', use_gpu=True)
    
    # Analyze by content type
    content_types = ['articles', 'social_media', 'technical_docs']
    for content_type in content_types:
        # Create figure using plotly
        fig = go.Figure()
        
        # Add performance data
        performance_data = analyzer.plot_pipeline_performance(
            model_filter=['gpt-4', 'gpt-3.5-turbo'],
            time_window=30,
            interactive=True
        )
        
        # Update layout with content type
        fig.update_layout(
            title=f"Performance Analysis - {content_type}",
            xaxis_title="Date",
            yaxis_title="Performance Metric",
            showlegend=True
        )
        
        # Save interactive plot
        fig.write_html(f"{content_type}_performance.html")
    
    print("Content generation pipeline analysis completed.")

def monitor_customer_support_quality():
    """Monitor quality of customer support responses.
    
    Use Case: Customer support team tracking response quality,
    consistency, and efficiency across different support tiers.
    """
    analyzer = LLMEnhancedAnalyzer(llm_backend='gpt', use_gpu=True)
    
    # Analyze by support tier
    support_tiers = ['tier1', 'tier2', 'tier3']
    for tier in support_tiers:
        # Quality analysis
        fig = analyzer.plot_quality_details(
            model_name='gpt-4',
            time_window=7
        )
        
        # Add support tier metrics
        fig.add_trace(
            go.Scatter(
                x=pd.date_range(end=datetime.now(), periods=7, freq='D'),
                y=np.random.normal(0.85, 0.05, 7),  # Example metrics
                name=f"{tier} satisfaction"
            )
        )
        
        fig.write_html(f"{tier}_support_quality.html")
    
    print("Customer support quality monitoring completed.")

def analyze_research_assistant_performance():
    """Analyze performance of research assistant pipeline.
    
    Use Case: Research team monitoring accuracy, depth, and
    efficiency of literature analysis and synthesis.
    """
    analyzer = LLMEnhancedAnalyzer(llm_backend='gpt', use_gpu=True)
    
    # Analysis parameters
    research_metrics = {
        'citation_accuracy': 0.95,
        'synthesis_depth': 0.88,
        'source_coverage': 0.92
    }
    
    # Detailed quality analysis
    fig = analyzer.plot_quality_details(
        model_name='gpt-4',
        time_window=30
    )
    
    # Add research-specific metrics
    for metric, score in research_metrics.items():
        fig.add_trace(
            go.Bar(
                x=[metric],
                y=[score],
                name=metric
            )
        )
    
    fig.write_html("research_assistant_performance.html")
    print("Research assistant performance analysis completed.")

def monitor_code_review_pipeline():
    """Monitor performance of automated code review pipeline.
    
    Use Case: Development team tracking code review quality,
    suggestion accuracy, and review time efficiency.
    """
    analyzer = LLMEnhancedAnalyzer(llm_backend='gpt', use_gpu=True)
    
    # Analyze code review metrics
    fig = analyzer.plot_pipeline_performance(
        time_window=14,
        interactive=True
    )
    
    # Add code review specific metrics
    review_metrics = {
        'suggestion_accuracy': np.random.normal(0.9, 0.05, 14),
        'false_positive_rate': np.random.normal(0.1, 0.02, 14),
        'review_time_efficiency': np.random.normal(0.85, 0.07, 14)
    }
    
    for metric, values in review_metrics.items():
        fig.add_trace(
            go.Scatter(
                x=pd.date_range(end=datetime.now(), periods=14, freq='D'),
                y=values,
                name=metric
            )
        )
    
    fig.write_html("code_review_performance.html")
    print("Code review pipeline monitoring completed.")

def analyze_translation_quality():
    """Analyze quality of translation pipeline.
    
    Use Case: Localization team monitoring translation quality,
    cultural accuracy, and consistency across languages.
    """
    analyzer = LLMEnhancedAnalyzer(llm_backend='gpt', use_gpu=True)
    
    # Analyze by language pair
    language_pairs = [
        ('en', 'es'), ('en', 'fr'), ('en', 'de'),
        ('es', 'en'), ('fr', 'en'), ('de', 'en')
    ]
    
    for source, target in language_pairs:
        fig = analyzer.plot_quality_details(
            model_name='gpt-4',
            time_window=30
        )
        
        # Add translation-specific metrics
        translation_metrics = {
            'semantic_accuracy': np.random.normal(0.88, 0.05, 30),
            'cultural_relevance': np.random.normal(0.85, 0.06, 30),
            'style_consistency': np.random.normal(0.9, 0.04, 30)
        }
        
        for metric, values in translation_metrics.items():
            fig.add_trace(
                go.Scatter(
                    x=pd.date_range(end=datetime.now(), periods=30, freq='D'),
                    y=values,
                    name=f"{source}-{target} {metric}"
                )
            )
        
        fig.write_html(f"translation_quality_{source}_{target}.html")
    
    print("Translation quality analysis completed.")

def monitor_educational_content_adaptation():
    """Monitor educational content adaptation pipeline.
    
    Use Case: Educational team tracking content adaptation quality,
    learning level appropriateness, and engagement metrics.
    """
    analyzer = LLMEnhancedAnalyzer(llm_backend='gpt', use_gpu=True)
    
    # Analyze by education level
    education_levels = ['elementary', 'middle_school', 'high_school', 'university']
    
    for level in education_levels:
        fig = analyzer.plot_pipeline_performance(
            time_window=30,
            interactive=True
        )
        
        # Add education-specific metrics
        education_metrics = {
            'age_appropriateness': np.random.normal(0.92, 0.03, 30),
            'concept_clarity': np.random.normal(0.88, 0.05, 30),
            'engagement_score': np.random.normal(0.85, 0.06, 30)
        }
        
        for metric, values in education_metrics.items():
            fig.add_trace(
                go.Scatter(
                    x=pd.date_range(end=datetime.now(), periods=30, freq='D'),
                    y=values,
                    name=f"{level} {metric}"
                )
            )
        
        fig.write_html(f"education_adaptation_{level}.html")
    
    print("Educational content adaptation monitoring completed.")

def main():
    """Run all pipeline visualization use cases."""
    print("Starting pipeline visualization use cases...")
    
    try:
        # Set up OpenAI key (replace with your key)
        SynergyMLConfig.set_openai_key("your-openai-key")
        
        # Run use cases
        analyze_content_generation_pipeline()
        monitor_customer_support_quality()
        analyze_research_assistant_performance()
        monitor_code_review_pipeline()
        analyze_translation_quality()
        monitor_educational_content_adaptation()
        
        print("\nAll pipeline visualization use cases completed successfully!")
        
    except Exception as e:
        print(f"Error running use cases: {str(e)}")

if __name__ == "__main__":
    main() 