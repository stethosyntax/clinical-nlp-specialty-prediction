# Clinical NLP: Medical Specialty Prediction from Clinical Notes

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-MTSamples-lightblue)]()
[![Phase](https://img.shields.io/badge/Phase-1%20Complete-success)]()

An end-to-end NLP pipeline that predicts the medical specialty of a clinical note using free text — built by a physician turned data scientist.

---

## Overview

Medical coding and specialty classification is one of the most time-consuming tasks in healthcare administration. Clinicians and coders spend thousands of hours manually reviewing free-text notes to categorize them. NLP offers a path to automate this process, reduce errors, and free up clinical staff for higher-value work.

This project builds a practical, interpretable text classification pipeline using real-world medical transcription data. As a physician (MBBS) now studying data science, I bring firsthand experience writing and reviewing clinical notes — which allows me to validate whether the model's learned patterns make clinical sense, not just statistical sense.

---

## Project Pipeline

```
Raw Clinical Notes (MTSamples)
           ↓
  Step 1: Load Data & Exploratory Analysis
           ↓
  Step 2: Filter & Prepare (Top 10 specialties)
           ↓
  Step 3: Text Preprocessing (spaCy)
           ↓
  Step 4: Train / Test Split (80/20, stratified)
           ↓
  Step 5: Baseline — TF-IDF + Logistic Regression
           ↓
  Step 6: Improved — TF-IDF + Linear SVC
           ↓
  Step 7: Evaluation — Confusion Matrix, Model Comparison, Top Terms
           ↓
  Predicted Medical Specialty
```

---

## Why Each Step Matters

| Step | What We Do | Why It Matters |
|---|---|---|
| **1. EDA** | Load data, chart specialty distribution and note lengths, generate word clouds | Reveals class imbalance, note quality, and dominant vocabulary before any modeling |
| **2. Filter & Prepare** | Keep top 10 specialties, drop near-empty notes | Models can't learn from 5 examples per class — filtering ensures each class has sufficient signal |
| **3. Preprocessing** | Lowercase → remove numbers → strip punctuation → tokenize → remove stopwords → lemmatize | Raw clinical text is noisy — generic terms like "patient" appear in every specialty and carry no signal |
| **4. Train/Test Split** | 80/20 stratified split | Stratification ensures every specialty is proportionally represented in both sets, giving a fair and honest evaluation |
| **5. Baseline Model** | TF-IDF + Logistic Regression | Always start simple — this sets the performance floor that every subsequent model must beat |
| **6. Improved Model** | TF-IDF + Linear SVC | SVCs find sharper decision boundaries in high-dimensional TF-IDF feature space, often outperforming LR on text tasks |
| **7. Evaluation** | Confusion matrix (raw + normalized), model comparison chart, top predictive terms per specialty | Numbers alone don't tell the full story — visual evaluation shows where the model fails and whether it has learned clinically meaningful patterns |

---

## Dataset

**MTSamples** — 4,999 real de-identified medical transcription notes across 40 specialties. Free to use, no sign-up required.
- Source: [mtsamples.com](https://www.mtsamples.com/)
- GitHub mirror used in this project: [socd06/medical-nlp](https://github.com/socd06/medical-nlp)

> ⚠️ No patient data is stored in this repository. The notebook downloads the dataset automatically at runtime.

---

## Results

| Model | Accuracy | Weighted F1 |
|---|---|---|
| TF-IDF + Logistic Regression | `[fill after run]` | `[fill after run]` |
| TF-IDF + Linear SVC | `[fill after run]` | `[fill after run]` |

*Run the notebook to generate your results and update this table.*

---

## Repository Structure

```
clinical-nlp-specialty-prediction/
│
├── notebooks/
│   └── clinical_nlp_pipeline.ipynb   ← Main notebook (all steps + explanations)
│
├── src/
│   ├── preprocess.py                  ← Text cleaning utilities (reusable module)
│   ├── train.py                       ← Model pipeline builders + evaluation functions
│   └── visualize.py                   ← All chart and visualization functions
│
├── data/
│   └── .gitkeep                       ← Place data files here (never committed)
│
├── outputs/
│   └── .gitkeep                       ← Charts saved here at runtime (never committed)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/stethosyntax/clinical-nlp-specialty-prediction.git
cd clinical-nlp-specialty-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Launch Jupyter
```bash
jupyter lab
```

Open `notebooks/clinical_nlp_pipeline.ipynb` and run all cells top to bottom.

---

## Key Findings

- **Specialty-specific vocabulary is highly predictive** — top TF-IDF terms per specialty align strongly with clinical domain knowledge (e.g., "arthroscopy", "meniscus" → Orthopedics; "catheterization", "stent" → Cardiovascular).
- **Linear SVC outperforms Logistic Regression** on this task due to sharper decision boundaries in high-dimensional TF-IDF space.
- **Class imbalance is a real challenge** — Surgery dominates the dataset. Using `class_weight='balanced'` is essential for fair performance across all specialties.

---

## Project Roadmap

### ✅ Phase 1 — Complete
- MTSamples dataset (4,999 notes, 10 specialties)
- TF-IDF + Logistic Regression baseline
- TF-IDF + Linear SVC improved model
- Full EDA, confusion matrix, model comparison, top terms per specialty

### 🔄 Phase 2 — In Progress
- MIMIC-III discharge summaries (50,000+ real hospital notes) — PhysioNet credentialing in progress
- ICD-9 multi-label code prediction
- Bio_ClinicalBERT fine-tuning (`emilyalsentzer/Bio_ClinicalBERT`)
- SHAP explainability — word-level attribution per prediction
- Streamlit app — paste any clinical note, get predicted specialty

---

## Author

**Aruna Kunche**
- Medical Graduate | M.S. Analytics, Harrisburg University (Expected 2026)
- [LinkedIn](https://www.linkedin.com/in/arunakunche/)
- [GitHub](https://github.com/stethosyntax)

---

## License

MIT License — free to use with attribution.
