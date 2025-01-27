"""Examples demonstrating cross-modal emotion analysis capabilities."""

import numpy as np
from pathlib import Path
from synergyml.multimodal.emotion import EmotionAnalyzer
from synergyml.multimodal.fusion import FusionVisualizer

def analyze_emotional_content(video_path: str):
    """Analyze emotional content across modalities.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_gpu=True)
    
    # Analyze emotional coherence
    results = analyzer.analyze_emotional_coherence(
        video_path,
        window_size=5,
        sampling_rate=16000
    )
    
    # Print summary
    print("\nEmotion Analysis Results:")
    print("-" * 50)
    
    print("\nOverall Coherence:", 
          results['emotion_alignment']['overall_coherence'])
    
    print("\nEmotion-wise Correlations:")
    for emotion, metrics in results['emotion_alignment']['correlation'].items():
        print(f"{emotion:10}: {metrics['coefficient']:.3f} "
              f"(p={metrics['p_value']:.3f})")
    
    print("\nTemporal Stability:")
    for emotion, stability in results['temporal_patterns']['stability'].items():
        print(f"{emotion:10}: {stability:.3f}")
    
    print("\nPeak Counts:")
    for emotion, peaks in results['temporal_patterns']['peaks'].items():
        print(f"{emotion:10}: Audio={len(peaks['audio'])} "
              f"Video={len(peaks['video'])}")
    
    return results

def analyze_emotion_dynamics(video_path: str):
    """Analyze emotion dynamics and transitions.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_gpu=True)
    visualizer = FusionVisualizer()
    
    # Get emotion analysis results
    results = analyzer.analyze_emotional_coherence(video_path)
    
    # Analyze transitions between emotions
    transitions = []
    timestamps = results['raw_emotions']['audio']['timestamps']
    
    for i in range(1, len(timestamps)):
        prev_emotions = {
            modality: max(
                results['raw_emotions'][modality]['emotions'].items(),
                key=lambda x: x[1][i-1]
            )[0]
            for modality in ['audio', 'video']
        }
        
        curr_emotions = {
            modality: max(
                results['raw_emotions'][modality]['emotions'].items(),
                key=lambda x: x[1][i]
            )[0]
            for modality in ['audio', 'video']
        }
        
        if prev_emotions != curr_emotions:
            transitions.append({
                'time': timestamps[i],
                'from': prev_emotions,
                'to': curr_emotions
            })
    
    # Print transition analysis
    print("\nEmotion Transitions:")
    print("-" * 50)
    
    for t in transitions:
        print(f"\nAt {t['time']:.2f}s:")
        print(f"Audio: {t['from']['audio']} -> {t['to']['audio']}")
        print(f"Video: {t['from']['video']} -> {t['to']['video']}")
    
    return transitions

def analyze_emotional_segments(video_path: str):
    """Analyze emotional segments and patterns.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_gpu=True)
    
    # Get emotion analysis results
    results = analyzer.analyze_emotional_coherence(video_path)
    
    # Find coherent emotional segments
    segments = []
    timestamps = results['raw_emotions']['audio']['timestamps']
    window_size = 5
    
    for emotion in analyzer.EMOTION_CATEGORIES:
        audio_signal = results['raw_emotions']['audio']['emotions'][emotion]
        video_signal = results['raw_emotions']['video']['emotions'][emotion]
        
        # Find segments where both modalities show strong emotion
        for i in range(window_size, len(timestamps) - window_size):
            audio_window = audio_signal[i-window_size:i+window_size]
            video_window = video_signal[i-window_size:i+window_size]
            
            if (np.mean(audio_window) > 0.5 and 
                np.mean(video_window) > 0.5):
                
                segments.append({
                    'start_time': timestamps[i-window_size],
                    'end_time': timestamps[i+window_size],
                    'emotion': emotion,
                    'intensity': np.mean([
                        np.mean(audio_window),
                        np.mean(video_window)
                    ])
                })
    
    # Merge overlapping segments
    segments.sort(key=lambda x: x['start_time'])
    merged_segments = []
    
    for segment in segments:
        if not merged_segments:
            merged_segments.append(segment)
        else:
            last = merged_segments[-1]
            if segment['start_time'] <= last['end_time']:
                # Merge if same emotion or higher intensity
                if (segment['emotion'] == last['emotion'] or
                    segment['intensity'] > last['intensity']):
                    last['end_time'] = max(last['end_time'], 
                                         segment['end_time'])
                    last['intensity'] = max(last['intensity'], 
                                          segment['intensity'])
            else:
                merged_segments.append(segment)
    
    # Print segment analysis
    print("\nEmotional Segments:")
    print("-" * 50)
    
    for segment in merged_segments:
        duration = segment['end_time'] - segment['start_time']
        print(f"\n{segment['emotion']:10} "
              f"({segment['start_time']:.2f}s - {segment['end_time']:.2f}s, "
              f"duration: {duration:.2f}s)")
        print(f"Intensity: {segment['intensity']:.3f}")
    
    return merged_segments

def analyze_emotion_complexity_example(video_path: str):
    """Analyze emotional complexity and dynamics.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_gpu=True)
    
    # Get emotion analysis results
    results = analyzer.analyze_emotional_coherence(video_path)
    complexity = analyzer.analyze_emotion_complexity(results['raw_emotions'])
    
    # Print complexity analysis
    print("\nEmotion Complexity Analysis:")
    print("-" * 50)
    
    for modality in ['audio', 'video']:
        print(f"\n{modality.upper()} Modality:")
        
        print("\nAverage Entropy:", 
              np.mean(complexity['entropy'][modality]))
        
        print("\nDominant Emotions:")
        for emotion, score in sorted(
            complexity['dominance'][modality].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"{emotion:10}: {score:.3f}")
        
        print("\nEmotion Transitions:")
        for t in complexity['transitions'][modality][:5]:  # Show first 5
            print(f"At {t['time']:.2f}s: {t['from']} -> {t['to']}")
        
        print("\nAverage Emotion Blending:", 
              np.mean(complexity['blending'][modality]))
    
    return complexity

def analyze_emotion_synchronization_example(video_path: str):
    """Analyze synchronization between audio and video emotions.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_gpu=True)
    
    # Get emotion analysis results
    results = analyzer.analyze_emotional_coherence(video_path)
    sync = analyzer.analyze_emotion_synchronization(results['raw_emotions'])
    
    # Print synchronization analysis
    print("\nEmotion Synchronization Analysis:")
    print("-" * 50)
    
    for emotion in analyzer.EMOTION_CATEGORIES:
        print(f"\n{emotion.upper()}:")
        
        # Lag correlation
        lag_corr = sync['lag_correlation'][emotion]
        max_idx = np.argmax(np.abs(lag_corr['correlations']))
        max_lag = lag_corr['lags'][max_idx]
        max_corr = lag_corr['correlations'][max_idx]
        
        print(f"Maximum Correlation: {max_corr:.3f} at lag {max_lag}")
        
        # Average coherence
        print(f"Average Coherence: {np.mean(sync['coherence'][emotion]):.3f}")
        
        # Average mutual information
        print(f"Average Mutual Information: "
              f"{np.mean(sync['mutual_information'][emotion]):.3f}")
    
    return sync

def analyze_emotion_context_example(video_path: str):
    """Analyze emotional context and patterns.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_gpu=True)
    
    # Get emotion analysis results
    results = analyzer.analyze_emotional_coherence(video_path)
    context = analyzer.analyze_emotion_context(results['raw_emotions'])
    
    # Print context analysis
    print("\nEmotion Context Analysis:")
    print("-" * 50)
    
    for modality in ['audio', 'video']:
        print(f"\n{modality.upper()} Modality:")
        
        print("\nEmotion Sequences:")
        for seq in context['emotion_sequences'][modality][:5]:  # Show first 5
            print(f"{seq['emotion']:10} for {seq['duration']:.2f}s")
        
        print("\nTemporal Patterns:")
        for emotion, pattern in context['temporal_patterns'][modality].items():
            if pattern['period'] > 0:
                print(f"{emotion:10}: Period={pattern['period']:.2f}s, "
                      f"Strength={pattern['strength']:.3f}")
    
    print("\nTop Emotion Co-occurrences:")
    co_occur = context['co_occurrence']
    for i in range(len(analyzer.EMOTION_CATEGORIES)):
        for j in range(i+1, len(analyzer.EMOTION_CATEGORIES)):
            if co_occur[i,j] > 0.1:  # Show significant co-occurrences
                print(f"{analyzer.EMOTION_CATEGORIES[i]} + "
                      f"{analyzer.EMOTION_CATEGORIES[j]}: "
                      f"{co_occur[i,j]:.3f}")
    
    return context

def analyze_emotion_changepoints_example(video_path: str):
    """Analyze emotion change points and regimes.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_gpu=True)
    
    # Get emotion analysis results
    results = analyzer.analyze_emotional_coherence(video_path)
    changepoints = analyzer.analyze_emotion_changepoints(
        results['raw_emotions'],
        window_size=5,
        threshold=0.3
    )
    
    # Print change point analysis
    print("\nEmotion Change Point Analysis:")
    print("-" * 50)
    
    for modality in ['audio', 'video']:
        print(f"\n{modality.upper()} Modality:")
        
        print("\nSignificant Change Points:")
        for cp in changepoints['changepoints'][modality]:
            print(f"\nAt {cp['time']:.2f}s (score: {cp['score']:.3f}):")
            print("Before:", end=" ")
            for emotion, prob in cp['before_state'].items():
                if prob > 0.1:  # Show significant emotions
                    print(f"{emotion}: {prob:.3f}", end=", ")
            print("\nAfter: ", end=" ")
            for emotion, prob in cp['after_state'].items():
                if prob > 0.1:  # Show significant emotions
                    print(f"{emotion}: {prob:.3f}", end=", ")
            print()
        
        print("\nRegime Statistics:")
        for i, regime in enumerate(changepoints['regime_statistics'][modality]):
            print(f"\nRegime {i+1}:")
            print(f"Duration: {regime['duration']:.2f}s")
            print(f"Dominant Emotion: {regime['dominant_emotion']}")
            print(f"Stability: {regime['stability']:.3f}")
            print(f"Complexity: {regime['complexity']:.3f}")
    
    return changepoints

def analyze_emotion_trends_example(video_path: str):
    """Analyze emotion trends and patterns.
    
    Parameters
    ----------
    video_path : str
        Path to video file
    """
    # Initialize analyzer
    analyzer = EmotionAnalyzer(use_gpu=True)
    
    # Get emotion analysis results
    results = analyzer.analyze_emotional_coherence(video_path)
    trends = analyzer.analyze_emotion_trends(results['raw_emotions'])
    
    # Print trend analysis
    print("\nEmotion Trend Analysis:")
    print("-" * 50)
    
    for modality in ['audio', 'video']:
        print(f"\n{modality.upper()} Modality:")
        
        print("\nSeasonal Patterns:")
        for emotion in trends['seasonality'][modality].keys():
            patterns = trends['seasonality'][modality][emotion]['patterns']
            print(f"\n{emotion}:")
            for i, pattern in enumerate(patterns, 1):
                print(f"Pattern {i}:")
                print(f"Frequency: {1/pattern['frequency']:.2f}s")
                print(f"Amplitude: {pattern['amplitude']:.3f}")
        
        print("\nMomentum Analysis:")
        for emotion in trends['momentum'][modality].keys():
            momentum = trends['momentum'][modality][emotion]
            print(f"\n{emotion}:")
            print(f"Trend Strength: {momentum['trend_strength']:.3f}")
            print(f"Average Rate of Change: "
                  f"{np.mean(np.abs(momentum['rate_of_change'])):.3f}")
            print(f"Average Acceleration: "
                  f"{np.mean(np.abs(momentum['acceleration'])):.3f}")
    
    return trends

def main():
    """Run emotion analysis examples."""
    # Example video file path
    video_path = "example_video.mp4"
    
    # Check if video file exists
    if not Path(video_path).exists():
        print(f"Please provide a valid video file path. Could not find: {video_path}")
        return
    
    print("Running Emotion Analysis Examples...")
    
    # Run basic examples
    print("\n1. Basic Emotional Content Analysis")
    content_results = analyze_emotional_content(video_path)
    
    print("\n2. Emotion Dynamics Analysis")
    dynamics_results = analyze_emotion_dynamics(video_path)
    
    print("\n3. Emotional Segment Analysis")
    segment_results = analyze_emotional_segments(video_path)
    
    # Run advanced examples
    print("\n4. Emotion Complexity Analysis")
    complexity_results = analyze_emotion_complexity_example(video_path)
    
    print("\n5. Emotion Synchronization Analysis")
    sync_results = analyze_emotion_synchronization_example(video_path)
    
    print("\n6. Emotion Context Analysis")
    context_results = analyze_emotion_context_example(video_path)
    
    print("\n7. Emotion Change Point Analysis")
    changepoint_results = analyze_emotion_changepoints_example(video_path)
    
    print("\n8. Emotion Trend Analysis")
    trend_results = analyze_emotion_trends_example(video_path)
    
    # Create visualizations
    from synergyml.multimodal.emotion.visualization import (
        plot_emotion_changepoints,
        plot_emotion_trends,
        plot_emotion_summary
    )
    
    # Plot change points
    fig_cp = plot_emotion_changepoints(changepoint_results)
    fig_cp.show()
    
    # Plot trends
    fig_trends = plot_emotion_trends(trend_results)
    fig_trends.show()
    
    # Plot comprehensive summary
    all_results = {
        'raw_emotions': content_results,
        'changepoints': changepoint_results,
        'trends': trend_results,
        'synchronization': sync_results,
        'context': context_results
    }
    fig_summary = plot_emotion_summary(all_results)
    fig_summary.show()
    
    print("\nAll emotion analysis examples completed successfully!")

if __name__ == "__main__":
    main() 