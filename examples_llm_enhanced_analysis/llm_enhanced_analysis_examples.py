"""Examples demonstrating LLM-enhanced multimodal analysis capabilities."""

from synergyml.multimodal.llm_integration.analyzer import LLMEnhancedAnalyzer
from synergyml.config import SynergyMLConfig
import os

def basic_llm_analysis(video_path: str):
    """Basic example of LLM-enhanced analysis."""
    # Initialize analyzer
    analyzer = LLMEnhancedAnalyzer(use_gpu=True)
    
    # Perform analysis
    results = analyzer.analyze_with_llm_understanding(video_path)
    
    # Print LLM insights
    print("\nLLM Analysis Insights:")
    print(results["llm_insights"])
    
    return results

def custom_prompt_analysis(video_path: str):
    """Example with custom analysis prompt."""
    analyzer = LLMEnhancedAnalyzer(use_gpu=True)
    
    # Custom prompt focusing on specific aspects
    custom_prompt = """
    Analyze the emotional patterns in this video focusing on:
    1. Emotional authenticity and congruence
    2. Potential emotional manipulation techniques
    3. Cultural and contextual influences
    4. Emotional resonance with target audience
    
    Provide specific examples and timestamps where possible.
    """
    
    results = analyzer.analyze_with_llm_understanding(
        video_path,
        analysis_prompt=custom_prompt
    )
    
    print("\nCustom Analysis Results:")
    print(results["llm_insights"])
    
    return results

def advanced_llm_analysis(video_path: str):
    """Advanced analysis combining multiple aspects."""
    analyzer = LLMEnhancedAnalyzer(
        model="gpt-4",  # Using GPT-4 for more sophisticated analysis
        use_gpu=True
    )
    
    # Get basic analysis
    results = analyzer.analyze_with_llm_understanding(video_path)
    
    # Get emotion analysis details
    emotion_data = results["emotion_analysis"]
    
    # Additional custom analysis focusing on patterns
    pattern_prompt = f"""
    Based on the emotional patterns detected:
    
    Audio Emotions: {emotion_data['raw_emotions']['audio']}
    Video Emotions: {emotion_data['raw_emotions']['video']}
    
    Please provide:
    1. A detailed analysis of emotional synchronization
    2. Identification of any emotional dissonance
    3. Analysis of emotional pacing and rhythm
    4. Recommendations for emotional optimization
    """
    
    pattern_results = analyzer.analyze_with_llm_understanding(
        video_path,
        analysis_prompt=pattern_prompt
    )
    
    # Combine insights
    results["pattern_analysis"] = pattern_results["llm_insights"]
    
    print("\nAdvanced Analysis Results:")
    print("Basic Insights:")
    print(results["llm_insights"])
    print("\nPattern Analysis:")
    print(results["pattern_analysis"])
    
    return results

def main():
    """Run all examples."""
    # Check for example video file
    video_path = "example_video.mp4"
    if not os.path.exists(video_path):
        print(f"Please provide a video file at {video_path}")
        return
        
    print("Running LLM-enhanced analysis examples...")
    
    # Run examples
    basic_results = basic_llm_analysis(video_path)
    custom_results = custom_prompt_analysis(video_path)
    advanced_results = advanced_llm_analysis(video_path)
    
    print("\nAll examples completed successfully!")

if __name__ == "__main__":
    main() 