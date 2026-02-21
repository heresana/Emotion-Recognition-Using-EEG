## Explainable EEG Emotion Recognition

**First Author:** Samiya Kamal  
**Co-Author:** Sana Tasneem  

**Conference**: IEEE IMPACT 2026 (Accepted & Presented)

This repository contains the official implementation of our paper: *Explainable EEG Emotion Recognition Using Multi-View Graph Transformers and SHAP*.

We propose a hybrid deep learning architecture that models both the temporal dynamics and spatial topology of EEG signals to predict Valence and Arousal (VA) states, while utilizing SHAP for attention-based interpretability.

## Workflow of The Project
```text
EEG Input (DEAP)
        │
        ▼
Preprocessing
  ├─ Channel Selection (32 Channels)
  ├─ 8-second Window Segmentation
  └─ 4-second Overlapping Stride
        │
        ▼
Feature Extraction
  ├─ Differential Entropy (θ, α, β, γ)
  ├─ Skewness
  └─ Kurtosis
        │
        ▼
Structured Tensor
  (Window × Channel × Band × Feature)
        │
        ▼
BiLSTM
  → Temporal Dependency Modeling
        │
        ▼
Multi-View Graph Transformer (MVGT)
  ├─ Spatial Graph Encoding
  ├─ Spectral Embedding
  └─ Multi-Head Self-Attention
        │
        ▼
Emotion Classification
  ├─ High / Low Valence
  └─ High / Low Arousal
  (Russell’s Circumplex Model)
        │
        ▼
Interpretability Layer
  └─ SHAP-Based Feature Attribution
```
## Data Availability
The [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) used in this study is a publicly available but restricted dataset. To run this code, you must independently request access from the original authors, sign the EULA, and place the downloaded data in the `data/raw/` directory.
