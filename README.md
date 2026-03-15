# Lending Club Loan Default Analysis

IS5126 Group 7 Project

## Overview

This project analyzes the Lending Club loan dataset (2007-2018 Q4) to predict loan defaults. We perform exploratory data analysis, feature engineering, and train machine learning models, with results presented through an interactive Streamlit dashboard.

## Dataset

- **Source**: [Lending Club via Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- **Accepted loans**: ~2.26M records, 150+ features
- **Rejected loans**: ~27.6M records
- **Target**: Loan default prediction (binary classification)

## Project Structure

```
landingclub_group7/
├── data/
│   ├── raw/                  # Original CSV files (git-ignored)
│   └── processed/            # Cleaned & engineered datasets
├── notebooks/                # Jupyter notebooks for analysis
├── src/                      # Reusable Python modules
│   ├── data_loader.py        # Data loading utilities
│   ├── preprocessing.py      # Feature engineering & cleaning
│   ├── visualization.py      # Plotting helpers
│   └── model.py              # Training & evaluation
├── models/                   # Saved model artifacts
├── figures/                  # Generated plots & charts
├── streamlit_app/
│   └── app.py                # Streamlit dashboard
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (requires Kaggle API key)
kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip

# Launch dashboard
streamlit run streamlit_app/app.py
```
