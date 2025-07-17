# AVFormer: Injecting Vision into Frozen Speech Models for Zero-Shot AV-ASR

**Paper:** https://arxiv.org/pdf/2303.16501  
**Code:** Code not found  
**Year:** 2023

## Summary

This paper addresses the significant challenge of creating robust and adaptable **Audiovisual Automatic Speech Recognition (AV-ASR)** systems. Traditional methods for training AV-ASR models require large, specialized audiovisual datasets, which are costly and difficult to obtain. Furthermore, these models often require extensive end-to-end training and struggle to generalize to new, unseen domains (a "zero-shot" setting)[2]. The core problem is how to effectively leverage visual information (like a speaker's context or environment) to improve speech recognition, especially in noisy conditions, without the prohibitive cost of retraining massive models from scratch[3].

To tackle this, the authors introduce **AVFormer**, a novel and lightweight method for augmenting existing, powerful, audio-only ASR models with visual capabilities. The key idea is to take a state-of-the-art, pre-trained ASR model and keep it "frozen," meaning its core parameters are not changed. Instead, visual information is injected into this model using small, trainable modules called **adapters**[3]. This approach is highly efficient, requiring minimal new parameters and only a small amount of weakly labeled video data for training[2]. A crucial innovation is a **two-phase curriculum learning strategy** that first adapts the model to the new audio domain and then integrates the visual information, ensuring the model learns to use both streams effectively.

### Architecture

The proposed method is built upon a frozen, state-of-the-art audio ASR model (BEST-RQ) and a frozen visual feature extractor (CLIP). The innovation lies in how visual information is integrated without altering these powerful base models.

The architecture consists of three main parts:
*   **Frozen Audio Model**: The backbone is the **BEST-RQ** model, a Conformer-based ASR system. Its weights remain unchanged, preserving its powerful, pre-trained speech recognition capabilities.
*   **Frozen Visual Encoder**: Visual features are extracted from video frames using a pre-trained **CLIP** encoder (ViT-L/14), which is known for its strong generalization abilities. Its weights are also frozen.
*   **Lightweight Trainable Modules**: These are the only new components that are trained:
    *   **Visual Projection Layer**: A simple linear layer that projects the extracted visual features into the same embedding space as the audio features, turning them into "visual tokens" that the ASR model can understand.
    *   **Adapters**: Small, bottlenecked feed-forward layers are inserted into each block of the frozen Conformer encoder. These adapters are trained to help the model adjust to new video domains and fuse the audio and visual tokens effectively.

## Datasets Used

AVFormer is trained using a small, weakly-labeled dataset and evaluated on multiple benchmarks to test its zero-shot generalization across different domains.

| Dataset Name | Task(s) | Data Size | Other Details |
| :--- | :--- | :--- | :--- |
| HowTo100M | Lightweight Finetuning | 5% of the dataset | Used weakly-labeled data (pseudo-transcripts from an ASR model) to train the adapters and visual projection layer. |
| How2 | Zero-Shot Evaluation | 300 hours | Instructional videos with user-provided captions, used to test performance on in-domain data. |
| VisSpeech | Zero-Shot Evaluation | 503 clips | A challenging AV-ASR benchmark with more background noise and diverse accents, designed to test robustness. |
| Ego4D | Zero-Shot Evaluation | ~49 hours (val set) | Egocentric videos of daily activities, used to test generalization to a completely different domain. |
| LibriSpeech | Zero-Shot Evaluation | 960 hours | An audio-only benchmark used to ensure the model did not lose its original ASR capabilities ("catastrophic forgetting"). |

## Experiments and Results

The authors evaluate AVFormer against state-of-the-art models, including AVATAR (a top AV-ASR model) and BEST-RQ (the base audio-only model). The primary metric is Word Error Rate (WER), where lower is better. Results demonstrate that AVFormer achieves superior zero-shot performance across multiple benchmarks while being significantly more data- and parameter-efficient.

The key zero-shot results are summarized below. Notably, AVFormer was trained on only 5% of the HowTo100M dataset, while the baselines were finetuned on 100% of it.

| Method | Modality | How2 (WER %) | VisSpeech (WER %) | Ego4D (WER %) | LibriSpeech (WER %) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| AVATAR  | A+V | 18.37% | 35.59% | 71.97% | 24.08% |
| BEST-RQ [4] | Audio | 15.32% | 16.69% | 68.34% | 5.60% |
| **AVFormer (Ours)** | **A+V** | **13.69%** | **16.60%** | **64.75%** | **4.36%** |

An important finding from ablation studies was that the **two-phase curriculum learning strategy is essential**. Without it, the model fails to properly utilize the visual information and performs worse than the audio-only baseline. The experiments also showed that using a small number of visual tokens (4 frames) and simple feed-forward adapters provided the best balance of performance and efficiency.

## Model Components

The **AVFormer** architecture is designed to efficiently inject visual context into a frozen speech recognition pipeline.

-   **Input Layer**: The model takes an audio spectrogram and a sequence of video frames as input.
-   **Parallel Encoding**:
    -   The audio spectrogram is processed by the frozen **Conformer Spectrogram Tokenizer** to create audio tokens.
    -   Simultaneously, the video frames are passed through the frozen **CLIP Encoder** to extract high-level visual features.
-   **Visual Projection**: The extracted visual features are fed into a trainable **Visual-to-Audio Projection Layer**. This lightweight layer maps the visual features into the same dimensional space as the audio tokens, making them compatible.
-   **Fusion and Adaptation**:
    -   The audio tokens and the newly created visual tokens are combined and sent to the **Frozen Conformer Encoder**.
    -   Within each block of the Conformer, a trainable **Bottleneck Adapter** fine-tunes the combined features, allowing the model to learn the relationship between the audio and visual streams and adapt to the specific video domain.
-   **Output Generation**: The final sequence of contextualized tokens is passed to the **Frozen Conformer RNN-T Decoder**, which generates the transcribed text.

## Implications and Future Work

The implications of this research are significant. AVFormer provides a practical and highly efficient blueprint for adapting large, pre-trained unimodal models for multimodal tasks. It demonstrates that it is possible to achieve state-of-the-art performance in AV-ASR without the need for massive labeled datasets or costly end-to-end retraining[3]. This parameter-efficient approach is increasingly important as foundation models continue to grow in size and complexity[3].

By successfully improving ASR in challenging and diverse environments (instructional, egocentric), this work opens up possibilities for more robust speech recognition in real-world applications, such as video captioning, meeting transcription, and assistive technologies. The paper establishes a strong case for using lightweight adapters and curriculum learning as a go-to strategy for multimodal model adaptation.

## BibTeX Citation

```bibtex
@misc{seo2023avformer,
      title={AVFormer: Injecting Vision into Frozen Speech Models for Zero-Shot AV-ASR}, 
      author={Paul Hongsuck Seo and Arsha Nagrani and Cordelia Schmid},
      year={2023},
      eprint={2303.16501},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

 https://arxiv.org/pdf/2303.16501.pdf
[2] https://openaccess.thecvf.com/content/CVPR2023/papers/Seo_AVFormer_Injecting_Vision_Into_Frozen_Speech_Models_for_Zero-Shot_AV-ASR_CVPR_2023_paper.pdf
[3] https://research.google/blog/avformer-injecting-vision-into-frozen-speech-models-for-zero-shot-av-asr/
[4] https://www.reddit.com/r/MachineLearning/comments/1gb74j6/r_paper_summaries_for_some_of_our_papers_that/
[5] https://arxiv.org/abs/2303.16501
[6] https://arxiv.org/abs/2306.04787
[7] https://direct.mit.edu/coli/article/39/2/267/1425/Automatically-Assessing-Machine-Summary-Content
[8] https://www.calhealthreport.org/2024/12/17/analysis-as-a-former-attorney-for-violence-survivors-heres-why-restorative-justice-gives-me-hope/
