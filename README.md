## Explainable EEG Emotion Recognition

**First Author:** Samiya Kamal  
**Co-Author:** Sana Tasneem  

**Conference**: IEEE IMPACT 2026 (Accepted & Presented)

This repository contains the official implementation of our paper: *Explainable EEG Emotion Recognition Using Multi-View Graph Transformers and SHAP*.

We propose a hybrid deep learning architecture that models both the temporal dynamics and spatial topology of EEG signals to predict Valence and Arousal (VA) states, while utilizing SHAP for attention-based interpretability.

## Workflow of The Project

EEG (DEAP Dataset)
        │
        ▼
Preprocessing
  ├─ Channel Selection (32 Channels)
  ├─ 8s Window Segmentation
  └─ 4s Overlapping Step
        │
        ▼
Feature Extraction
  ├─ Differential Entropy (θ, α, β, γ bands)
  ├─ Skewness
  └─ Kurtosis
        │
        ▼
Structured Tensor (Epoch × Channel × Band × Feature)
        │
        ▼
BiLSTM (Temporal Modeling)
        │
        ▼
Multi-View Graph Transformer
  ├─ Spatial Graph Encoding
  ├─ Spectral Embedding
  └─ Self-Attention
        │
        ▼
Classification via Russell’s Circumplex Model
  ├─ Binarized High/Low Valence
  └─ Binarized High/Low Arousal
        │
        ▼
Attention-Based Interpretability (SHAP)

## Data Availability
The [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) used in this study is a publicly available but restricted dataset. To run this code, you must independently request access from the original authors, sign the EULA, and place the downloaded data in the `data/raw/` directory.
