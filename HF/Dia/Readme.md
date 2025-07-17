# Dia-1.6B by Nari Labs

**Model:** Dia-1.6B
**Repository:** [https://huggingface.co/nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626)  
**Github:** [https://github.com/nari-labs/dia](https://github.com/nari-labs/dia)
**License:** Apache 2.0

## Overview

**Dia-1.6B** is a fully open, **1.6 billion parameter text-to-speech (TTS) model** developed by Nari Labs. It is designed to generate highly realistic, expressive dialogue directly from a text transcript. Unlike traditional TTS systems that simply read text, Dia can perform a script, incorporating different speakers, emotional tones, and non-verbal sounds like laughter or coughing.

The model is built on PyTorch and is available on Hugging Face, where it can be used with the `transformers` library. Currently, it only supports English generation. A demo is available on Hugging Face Spaces for users to test its capabilities.

## Key Features

Dia offers a range of features that make it a powerful tool for generating dynamic and natural-sounding audio:

*   **Dialogue Generation**: The model can produce back-and-forth conversations by recognizing speaker tags like `[S1]` and `[S2]` in the input text.
*   **Non-Verbal Communication**: It can interpret and generate various non-verbal sounds from text prompts, such as `(laughs)`, `(coughs)`, `(sighs)`, and `(singing)`, adding a layer of realism to the audio.
*   **Voice Cloning and Tone Control**: Users can condition the output on a provided audio sample. This allows the model to clone the voice and mimic the tone and emotion of the prompt audio for the new generated content.
*   **Open Weights**: The model is released with open weights under the Apache 2.0 license, promoting transparency and community collaboration in voice technology research.

## Usage and Implementation

Dia can be integrated into projects as a Python library or directly through the Hugging Face `transformers` library.

### Basic Text-to-Speech

Here is an example of generating dialogue from a simple text script:
```python
import soundfile as sf
from dia.model import Dia

# Load the model
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626")

# Provide a script with speaker tags and non-verbal cues
text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."

# Generate the audio
output = model.generate(text)

# Save the output file
sf.write("dialogue.mp3", output, 44100)
```

### Voice Cloning with an Audio Prompt

This example demonstrates how to use an existing audio file to guide the voice and tone of the generated speech:
```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor, DiaForConditionalGeneration

torch_device = "cuda"
model_checkpoint = "nari-labs/Dia-1.6B-0626"

# Load an audio dataset for the prompt
ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=44100))
audio_prompt = ds[-1]["audio"]["array"]

# The text includes a transcript of the prompt audio followed by the new text to generate
text = "[S1] I know. It's going to save me a lot of money, I hope. [S2] I sure hope so for you."

# Process inputs
processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, audio=audio_prompt, padding=True, return_tensors="pt").to(torch_device)
prompt_len = processor.get_audio_prompt_len(inputs["decoder_attention_mask"])

# Load model and generate
model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(**inputs, max_new_tokens=256)

# Decode and save the newly generated part of the audio
outputs = processor.batch_decode(outputs, audio_prompt_len=prompt_len)
processor.save_audio(outputs, "cloned_voice_output.wav")
```

## Hardware and Performance

*   **GPU Requirement**: Dia has been tested on GPUs and requires PyTorch 2.0+ and CUDA 12.6. CPU support is planned for the future.
*   **VRAM**: The full model requires approximately **10GB of VRAM** to run.
*   **Inference Speed**: On an NVIDIA A4000 GPU, the model generates about 40 tokens per second (where 86 tokens roughly equals one second of audio). Performance can be improved on supported GPUs by using `torch.compile`.
*   **Quantization**: The team plans to release a quantized version for greater memory efficiency in the future.

## Ethical Considerations

The developers have outlined a strict disclaimer regarding the use of the model. The following uses are **strictly forbidden**:
*   Producing audio that misuses a real person's identity without their consent.
*   Generating deceptive or misleading content, such as fake news.
*   Any use that is illegal or intended to cause harm.
