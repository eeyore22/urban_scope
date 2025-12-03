# Pytorch implementation for evaluating VLM on a synthetic dataset
This repository covers pytorch implementation for evaluating and fine-tuning BLIP2, InstructBLIP and LLaVA-1.5 on a synthetic instruction dataset. 
This study was published at ICCV 2025 Workshop (Multimodal Reasoning and Slow Thinking in Large Model Era: Towards System 2 and Beyond).

## Preliminary
While VLMs are generally strong at recognizing what objects are present in an image, they often struggle to interpret where those objects are located and how they are spatially arranged.
Urban scenes contain repetitive elements such as sky, buildings, and greenery, but meaningful understanding depends on fine-grained spatial cues—relative proportions, occlusion (e.g., how much buildings block the sky), depth structure, and left–right layout.
These spatial relationships are crucial for urban planning [1], walkability assessment [2], and urban heat-island analysis [3], yet existing models rarely reason over such cues in a systematic way.
Therefore, we 1) constructed a synthetic dataset tailored to urban scenes and relevant question-answer pairs and 2) ran zero-shot evaluation and fine-tuning on representative open-source VLMs to test whether domain-specific instruction tuning using a synthetic dataset can help close the domain gap.

![Domain Gap](intro_figure.png)

## Dependency
<pre> '''pip install requirements.txt''' </pre>
