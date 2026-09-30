# SHL Grammar Scoring Engine

A Grammar Scoring Engine that evaluates spoken English grammar quality from audio recordings. Built for the [SHL Hiring Assessment 2026](https://www.kaggle.com/competitions/shl-hiring-assessment-2026) Kaggle challenge.

## Problem

Given a spoken audio sample (45–60 seconds, `.wav`), predict a continuous grammar score from **0 to 5** based on the MOS Likert Grammar Score rubric.

## Approach

1. **Speech-to-Text**: Transcribe audio using OpenAI Whisper
2. **Feature Engineering**: Extract grammar-quality features from transcripts
3. **Regression Model**: Predict continuous grammar scores

See the [notebook](notebook.ipynb) for the full pipeline, analysis, and results.

## Dataset

- **Training**: 769 audio samples with grammar scores
- **Testing**: 216 audio samples (predictions required)
- **Evaluation**: Pearson Correlation + RMSE

> Dataset is not included in this repo. Download from the Kaggle competition page.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── notebook.ipynb          # Main solution notebook (pipeline + report)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── Dataset_Final/          # Dataset (not tracked in git)
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    ├── train/              # Training audio files
    └── test/               # Test audio files
```

## Results

| Metric | Score |
|---|---|
| **Train RMSE** | TBD |
| **Pearson Correlation** | TBD |

## Author

Saurav Singh
