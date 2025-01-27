"""Prompt templates for vision tasks."""

# Image classification prompts
BASIC_IMAGE_CLASSIFICATION = """
Classify this image into one of the following categories: {categories}
Respond with ONLY the category label and nothing else.
"""

DETAILED_IMAGE_CLASSIFICATION = """
Analyze this image carefully and classify it into one of these categories: {categories}

Consider the following aspects:
1. Main subject and composition
2. Visual characteristics
3. Distinguishing features
4. Context and setting

Respond with ONLY the category label and nothing else.
"""

# Multimodal classification prompts
BASIC_MULTIMODAL_CLASSIFICATION = """
Analyze this image and text combination and assign relevant labels from: {categories}
You can assign up to {max_labels} labels.
Respond with ONLY the labels as a comma-separated list.

Text: {text}
"""

DETAILED_MULTIMODAL_CLASSIFICATION = """
Analyze this product review consisting of both text and image:

Text Review: {text}

Consider the following aspects:
1. Overall sentiment (positive/negative)
2. Product quality assessment
3. Visual accuracy compared to description
4. Specific features mentioned
5. Any discrepancies between text and image

Available labels: {categories}
You can assign up to {max_labels} labels.

Respond with ONLY the most relevant labels as a comma-separated list.
"""

# Image captioning prompts
BASIC_CAPTION = """
Provide a concise caption for this image.
"""

DETAILED_CAPTION = """
Provide a detailed description of this image, including:
1. Main subjects and their actions
2. Setting and environment
3. Notable details and features
4. Overall mood or atmosphere
"""

TECHNICAL_ANALYSIS = """
Analyze this image in detail, including:
1. Main subjects and their relationships
2. Setting and context
3. Notable visual elements and composition
4. Mood and atmosphere
5. Any text or symbols present
6. Technical aspects:
   - Lighting conditions
   - Color palette
   - Focus and depth
   - Composition techniques
"""

# Object detection prompts
OBJECT_DETECTION = """
List all significant objects visible in this image.
Format as a comma-separated list.
"""

SPATIAL_ANALYSIS = """
Analyze the spatial relationships between objects in this image.
Format each relationship as:
- [Object 1] is [relationship] [Object 2]
"""

# Scene understanding prompts
SCENE_ANALYSIS = """
Analyze this scene and describe:
1. Setting and environment
2. Main activities or events
3. Notable visual elements
4. Time of day/lighting conditions
5. Overall atmosphere

Format as bullet points.
"""

# Visual QA templates
VISUAL_QA = """
Question about the image: {question}

Analyze the image carefully and provide a clear, concise answer.
Focus only on visual elements that are relevant to the question.
"""

# Image comparison templates
IMAGE_COMPARISON = """
Compare these images in terms of:
1. Subject matter
2. Composition
3. Mood and atmosphere
4. Color palette
5. Technical execution

Highlight key similarities and differences.
""" 