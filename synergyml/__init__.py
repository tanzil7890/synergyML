"""SynergyML: Seamlessly integrate LLMs with scikit-learn."""

__version__ = '1.0.1'
__author__ = 'Mohammad Tanzil Idrisi'

from synergyml.config import SynergyMLConfig
from synergyml.classification import (
    ZeroShotGPTClassifier,
    MultiLabelZeroShotGPTClassifier,
    CoTGPTClassifier,
    FewShotGPTClassifier,
    DynamicFewShotGPTClassifier,
    MultiLabelFewShotGPTClassifier,
)
from synergyml.text2text import (
    GPTSummarizer,
    GPTTranslator,
)
from synergyml.multimodal import (
    ImageAnalyzer,
    ImageClassifier,
    MultimodalClassifier,
    ImageCaptioner,
    VideoAnalyzer,
    VideoClassifier,
    VideoActionRecognizer,
    VideoCaptioner,
    VideoSceneSegmenter,
    VideoTemporalAnalyzer,
    VideoObjectTracker,
    VideoEmotionAnalyzer,
)

__all__ = [
    'SynergyMLConfig',
    # Text Classification
    'ZeroShotGPTClassifier',
    'MultiLabelZeroShotGPTClassifier',
    'CoTGPTClassifier',
    'FewShotGPTClassifier',
    'DynamicFewShotGPTClassifier',
    'MultiLabelFewShotGPTClassifier',
    # Text Processing
    'GPTSummarizer',
    'GPTTranslator',
    # Vision and Multimodal
    'ImageAnalyzer',
    'ImageClassifier',
    'MultimodalClassifier',
    'ImageCaptioner',
    # Video Analysis
    'VideoAnalyzer',
    'VideoClassifier',
    'VideoActionRecognizer',
    'VideoCaptioner',
    'VideoSceneSegmenter',
    'VideoTemporalAnalyzer',
    'VideoObjectTracker',
    'VideoEmotionAnalyzer',
]
