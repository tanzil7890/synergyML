"""Examples demonstrating the pipeline visualization capabilities of SynergyML."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from synergyml.multimodal.llm_integration import LLMEnhancedAnalyzer
from synergyml.config import SynergyMLConfig

def basic_pipeline_visualization():
    """Demonstrate basic pipeline performance visualization."""
    # Initialize analyzer with default settings
    analyzer = LLMEnhancedAnalyzer(
        llm_backend='gpt',
        use_gpu=True,
        memory_backend='annoy'
    )
    
    # Plot overall pipeline performance
    fig = analyzer.plot_pipeline_performance(
        time_window=30,  # Last 30 days
        interactive=True
    )
    
    # Save the visualization
    fig.write_html("pipeline_performance.html")
    
    print("Basic pipeline visualization completed. Check pipeline_performance.html")

def detailed_model_analysis():
    """Demonstrate detailed quality analysis for specific models."""
    analyzer = LLMEnhancedAnalyzer(
        llm_backend='gpt',
        use_gpu=True
    )
    
    # Analyze quality metrics for GPT-4
    fig = analyzer.plot_quality_details(
        model_name='gpt-4',
        time_window=7  # Last week
    )
    
    fig.write_html("gpt4_quality_analysis.html")
    
    # Compare with GPT-3.5
    fig = analyzer.plot_quality_details(
        model_name='gpt-3.5-turbo',
        time_window=7
    )
    
    fig.write_html("gpt35_quality_analysis.html")
    
    print("Detailed model analysis completed. Check *_quality_analysis.html files")

def filtered_performance_analysis():
    """Demonstrate filtered pipeline performance analysis."""
    analyzer = LLMEnhancedAnalyzer(
        llm_backend='gpt',
        use_gpu=True
    )
    
    # Filter by model and minimum quality
    fig = analyzer.plot_pipeline_performance(
        model_filter=['gpt-4', 'gpt-3.5-turbo'],
        min_quality=0.8,
        interactive=True
    )
    
    fig.write_html("filtered_performance.html")
    
    print("Filtered performance analysis completed. Check filtered_performance.html")

def quality_metrics_exploration():
    """Demonstrate exploration of quality metrics."""
    analyzer = LLMEnhancedAnalyzer(
        llm_backend='gpt',
        use_gpu=True
    )
    
    # Plot quality metrics for all models
    fig = analyzer.plot_pipeline_performance(
        time_window=14,  # Last two weeks
        interactive=True
    )
    
    # Focus on quality metrics sections
    fig.update_layout(
        annotations=[
            dict(
                text="Focus on quality metrics distribution and radar plot",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.05,
                showarrow=False
            )
        ]
    )
    
    fig.write_html("quality_metrics_exploration.html")
    
    print("Quality metrics exploration completed. Check quality_metrics_exploration.html")

def performance_trend_analysis():
    """Demonstrate analysis of performance trends over time."""
    analyzer = LLMEnhancedAnalyzer(
        llm_backend='gpt',
        use_gpu=True
    )
    
    # Analyze long-term trends
    fig = analyzer.plot_pipeline_performance(
        time_window=90,  # Last 90 days
        interactive=True
    )
    
    # Add trend annotations
    fig.add_annotation(
        text="Observe long-term quality and performance trends",
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.05,
        showarrow=False
    )
    
    fig.write_html("performance_trends.html")
    
    print("Performance trend analysis completed. Check performance_trends.html")

def main():
    """Run all pipeline visualization examples."""
    print("Starting pipeline visualization examples...")
    
    try:
        # Set up OpenAI key (replace with your key)
        SynergyMLConfig.set_openai_key("your-openai-key")
        
        # Run examples
        basic_pipeline_visualization()
        detailed_model_analysis()
        filtered_performance_analysis()
        quality_metrics_exploration()
        performance_trend_analysis()
        
        print("\nAll pipeline visualization examples completed successfully!")
        print("\nGenerated HTML files:")
        print("- pipeline_performance.html")
        print("- gpt4_quality_analysis.html")
        print("- gpt35_quality_analysis.html")
        print("- filtered_performance.html")
        print("- quality_metrics_exploration.html")
        print("- performance_trends.html")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")

if __name__ == "__main__":
    main() 