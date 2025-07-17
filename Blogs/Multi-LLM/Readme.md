# Understanding Multimodal LLMs

**Paper:** https://sebastianraschka.com/blog/2024/understanding-multimodal-llms.html
**Year:** 2024

## Summary

This article addresses the fundamental challenge of **integrating multiple data types (modalities)**, such as text, images, sound, and video, into Large Language Models (LLMs). While traditional LLMs excel at processing text, extending their capabilities to understand and generate responses based on diverse inputs is crucial for more comprehensive AI applications. The core problem lies in effectively combining these disparate data formats within a unified model architecture.

To tackle this, the article explores two primary approaches to building multimodal LLMs: the **Unified Embedding Decoder Architecture (Method A)** and the **Cross-Modality Attention Architecture (Method B)**. The core idea behind these methods is to convert non-textual inputs into a format compatible with LLMs, either by aligning their embeddings and concatenating them as input or by integrating them through cross-attention mechanisms. These methods are distinct from traditional unimodal LLMs because they enable the processing of multiple input types, paving the way for applications like image captioning and information extraction from visual documents.

### Architecture

The article details two main architectural approaches for multimodal LLMs:

*   **Unified Embedding Decoder Architecture (Method A)**: This method uses a single decoder model, similar to a standard text-only LLM. Images are converted into tokens with the same embedding size as text tokens, allowing the LLM to process both types of input together after concatenation. This typically involves an image encoder (often a pre-trained Vision Transformer or ViT) that divides images into patches, encodes them, and then a linear projection layer (sometimes called a projector, adapter, or connector) that transforms these visual embeddings to match the LLM's text embedding dimensions.
*   **Cross-Modality Attention Architecture (Method B)**: This approach employs a cross-attention mechanism to integrate image and text embeddings directly within the attention layer. While still using an image encoder, instead of concatenating image tokens with text tokens as direct input to the LLM, the visual features are introduced via a cross-attention layer within the Transformer blocks. This allows the LLM to attend to relevant visual information while processing the text.

Both methods commonly start with a pre-trained, instruction-tuned text-only LLM as the backbone. The image encoder (e.g., CLIP, OpenCLIP) is often kept frozen, while a projector (a linear layer or small multi-layer perceptron) is trained to align visual features with the LLM's embedding space.

## Datasets Used

The article reviews several recent multimodal models, many of which utilize various datasets for training and pre-training. While specific detailed tables for each model's dataset use are not provided in the article, it broadly mentions:

| Dataset Name | Task(s) | Data Size | Other Details |
| :--- | :--- | :--- | :--- |
| Image-Text pairs | Pretraining vision encoders and multimodal LLMs | 2.5 billion (for Llama 3.2 ViT) | Used for training image encoders from scratch, often before connecting to the LLM. |
| Publicly available multimodal datasets | Training and fine-tuning various multimodal models | Varied (e.g., HowTo100M, COCO, Flickr30k) | Used for instruction fine-tuning and evaluating models across different tasks. |

## Experiments and Results

The article discusses the trade-offs and findings from various recent multimodal LLM implementations, rather than presenting a single consolidated experiment. It highlights general observations from several models:

| Method/Model Approach | Noted Advantage | Noted Disadvantage/Consideration |
| :--- | :--- | :--- |
| **Unified Embedding Decoder (Method A)** | Easier to implement, no LLM architecture modification needed  | Can overload input context with image tokens, potentially impacting computational efficiency  |
| **Cross-Modality Attention (Method B)** | More computationally efficient for high-resolution images, maintains text-only performance if LLM parameters are frozen  | Adds substantial parameters with cross-attention layers  |
| **Hybrid (e.g., NVLM-H)** | Combines advantages of both methods  | Complexity of combining two approaches  |

Key findings from the reviewed models include:

*   **Llama 3.2** uses a cross-attention approach and focuses on updating the image encoder while freezing the LLM to preserve text-only capabilities. It pre-trained a Vision Transformer (ViT-H/14) from scratch on 2.5 billion image-text pairs.
*   **Molmo** (Multimodal Open Language Model) and **PixMo** (Pixels for Molmo) aim for open-source model weights, dataset, and code. Molmo streamlines training by updating all parameters (LLM, connector, image encoder) in a unified approach.
*   **NVIDIA's NVLM** explores both Method A (NVLM-D) and Method B (NVLM-X), plus a hybrid (NVLM-H). They found NVLM-X superior for computational efficiency with high-res images, NVLM-D better for OCR, and NVLM-H combining both. They used Qwen2-72B-Instruct as the backbone LLM.
*   **Qwen2-VL** introduces a "Naive Dynamic Resolution" mechanism for handling variable image resolutions, using a modified ViT with 2D-RoPE positional embeddings.
*   **Pixtral 12B** (Mistral AI's first multimodal model) also uses Method A and trains its image encoder from scratch, supporting native variable image sizes.
*   **MM1.5** focuses on Method A and provides practical tips and ablation studies on data mixtures and coordinate tokens.
*   **Aria** is a mixture-of-experts model based on a cross-attention approach, pre-training both the LLM backbone and vision encoder.
*   **Baichuan-Omni** is a 7B parameter multimodal LLM based on Method A, with a three-stage training process involving sequential unfreezing of projector, vision encoder, and LLM. It uses the SigLIP vision encoder.
*   **Emu3** is a transformer-based decoder for image generation, trained from scratch and using DPO for human preference alignment.
*   **Janus** unifies multimodal understanding and generation within a single LLM by decoupling visual encoding pathways, using SigLIP for understanding and a VQ tokenizer for generation.

## Model Components

Multimodal LLM architectures generally involve three main components:

*   **Image Encoder**: This component processes raw image data to extract meaningful visual features. It typically divides images into smaller patches, which are then encoded by a Vision Transformer (ViT) or similar architecture. The image encoder's output is a sequence of visual embeddings.
*   **Projector (or Adapter/Connector)**: A linear layer or a small multi-layer perceptron (MLP) that maps the visual embeddings from the image encoder into a dimension compatible with the LLM's text token embeddings. Its role is to align the visual and textual feature spaces.
*   **Large Language Model (LLM) Backbone**: A pre-trained text-only LLM (e.g., Llama, Qwen) that forms the core of the multimodal model. Depending on the architecture (Unified Embedding Decoder or Cross-Modality Attention), the visual information is either concatenated with text tokens as direct input to the LLM or integrated via cross-attention layers within the LLM's Transformer blocks.

## Implications and Future Work

The implications of this research are substantial, demonstrating that multimodal LLMs can be successfully built in various ways, enabling AI systems to understand and generate content across different data types. This work opens up new possibilities for applications such as detailed image captioning, complex visual question answering, and even generating images from text. It lays the groundwork for future research into more efficient architectures, robust training methodologies, and broader integration of diverse modalities. The ongoing development of these models points towards a future where AI can perceive and interact with the world in a more holistic and human-like manner.
