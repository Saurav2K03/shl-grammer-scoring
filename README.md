# SHL Grammar Scoring Engine

A Grammar Scoring Engine that evaluates spoken English grammar quality from audio recordings. Built for the [SHL Hiring Assessment 2026](https://www.kaggle.com/competitions/shl-hiring-assessment-2026) Kaggle challenge.

## Problem

Given a spoken audio sample (45–60 seconds, `.wav`), predict a continuous grammar score from **0 to 5** based on the MOS Likert Grammar Score rubric.

## Approach

```
Audio (.wav) → Whisper ASR → Transcript + Metadata → Feature Engineering → XGBoost → Score (0-5)
```

### Pipeline

1. **Speech-to-Text**: Transcribe all 985 audio files using OpenAI Whisper
2. **Feature Extraction**: Extract **~157 features** per sample
   - **52 text features**: grammar errors (LanguageTool), readability scores (Flesch-Kincaid, ARI), vocabulary sophistication, sentence/clause structure, discourse markers, fluency indicators
   - **43 audio features**: MFCCs (13 × mean + std), pitch, energy (RMS), speaking rate, pause patterns, spectral centroid, zero crossing rate
   - **12 Whisper metadata**: model confidence (avg_logprob), no-speech probability, compression ratio, decoding temperature
   - **50 PCA embeddings**: MiniLM-L6-v2 sentence embeddings reduced from 384-dim
3. **Model**: XGBoost regressor with strong regularization (5-fold CV)

### Key Design Decisions

- **Single XGBoost > Ensemble**: With only 769 training samples, blending multiple models added noise. A single well-regularized XGBoost outperformed all blend strategies on the leaderboard.
- **PCA embeddings as direct features**: Feeding PCA-reduced sentence embeddings directly into XGBoost (instead of blending predictions from embedding-only models) improved generalization.
- **Whisper metadata as features**: avg_logprob, no_speech_prob, and temperature capture transcription difficulty, which is a proxy for speech clarity.

## Results

**Leaderboard RMSE: 0.5570**

*Cross-validation on training set (769 samples):*

| Model | CV RMSE | CV Pearson R |
|---|---|---|
| **XGBoost** | **0.6904** | **0.8307** |
| Ridge (baseline) | 0.7126 | 0.8179 |

## Dataset

- **Training**: 769 audio samples with grammar scores (0–5)
- **Testing**: 216 audio samples (predictions required)
- **Evaluation**: RMSE (leaderboard)

> Dataset is not included in this repo. Download from the [Kaggle competition page](https://www.kaggle.com/competitions/shl-hiring-assessment-2026).

## Project Structure

```
├── notebook.ipynb            # Main notebook: full pipeline end-to-end
├── feature_extraction.py     # Feature extraction module (~107 features)
├── transcribe_audio.py       # Whisper transcription script
├── requirements.txt          # Python dependencies
├── README.md
│
├── transcripts/              # Whisper output (not tracked)
│   ├── train_transcripts.json
│   └── test_transcripts.json
│
└── Dataset_Final/            # Dataset (not tracked)
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    ├── train/ (769 .wav files)
    └── test/ (216 .wav files)
```

## Setup

```bash
# Create conda environment
conda create -n shl python=3.10 -y
conda activate shl

# Install dependencies
pip install -r requirements.txt

# Transcribe audio files
KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1 python transcribe_audio.py

# Run the notebook
jupyter notebook notebook.ipynb
```

## Author

Saurav Singh
