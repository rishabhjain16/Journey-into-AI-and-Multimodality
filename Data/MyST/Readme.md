# MyST Children's Conversational Speech

**Paper:** [https://arxiv.org/pdf/2309.13347](https://arxiv.org/pdf/2309.13347)      
**Data-Path:** [https://catalog.ldc.upenn.edu/LDC2021S05](https://catalog.ldc.upenn.edu/LDC2021S05)      
**Year:** 2021      

## Summary

This dataset addresses the significant challenge of a lack of large-scale, conversational speech data from children for developing and testing speech technologies. Standard speech recognition and dialogue systems are often trained on adult speech, and perform poorly on children's speech due to differences in vocal tract size, pitch, and speaking patterns.

To tackle this, the **MyST (My Science Tutor) Children's Conversational Speech** dataset provides a substantial corpus of audio from children in grades 3-5 engaging in educational dialogues. The core idea was to collect natural speech by having students converse with a virtual science tutor about topics they had just studied in class. This method is distinct from read-speech corpora because it captures more spontaneous, interactive, and on-topic conversations, which are invaluable for training robust real-world systems.

### Data Collection

The data was gathered by having 1,371 students interact with a virtual tutor for 15-20 minute sessions after completing science investigations. The tutor asked open-ended questions related to on-screen media, prompting spoken answers from the students. This process was part of the research-based Full Option Science System (FOSS) curriculum, ensuring the conversations were context-rich and educationally relevant.

## Dataset Specifications

| Specification | Details |
| :--- | :--- |
| **Total Audio** | ~470 hours |
| **Speakers** | 1,371 students (Grades 3-5) |
| **Total Utterances** | 227,567 |
| **Transcribed Utterances** | 102,433 (~45%) |
| **Language** | English |
| **Audio Format** | Single channel, 16kHz, 16-bit FLAC |
| **Transcription Format** | UTF-8 plain text |

## Dataset Components

The **MyST** dataset is organized for use in speech technology research and development, particularly for Automatic Speech Recognition (ASR) systems.

*   **Speech Data**: Consists of ~470 hours of audio recordings from 10,496 individual sessions. The audio is provided in a standard, high-quality format suitable for acoustic modeling.
*   **Transcripts**: Contains time-aligned transcriptions for roughly 45% of the utterances. The transcription process followed rich guidelines, especially for data collected in the first phase, capturing details beyond just the words spoken.
*   **Data Partitions**: The dataset is pre-divided into **development, test, and train** sets, which is a standard practice that facilitates fair and reproducible ASR experiments.
*   **Pronunciation Dictionary**: A supplementary dictionary is included to aid in phonetics research and the development of pronunciation models.

## Implications and Applications

The release of this dataset has substantial implications for the field of speech technology. By providing a large, high-quality corpus of children's conversational speech, it enables significant advancements in several areas:

*   **Speech Recognition**: Training more accurate ASR systems for children, which can be used in educational software, toys, and accessibility tools.
*   **Spoken Dialogue Systems**: Building more natural and effective conversational agents and tutors for young users.
*   **Phonetics and Linguistics**: Researching the acoustic and linguistic properties of child speech and language development.
*   **Machine Reading**: Developing systems that can read and comprehend content alongside a child, using their speech for interaction.

## BibTeX Citation

```bibtex
@data{pradhan2021myst,
      title={MyST Children's Conversational Speech}, 
      author={Pradhan, Sameer and Cole, Ronald and Ward, Wayne},
      year={2021},
      publisher={Linguistic Data Consortium},
      address={Philadelphia},
      url={https://doi.org/10.35111/cyxy-p432},
      note={LDC catalog number LDC2021S05}
}
```
