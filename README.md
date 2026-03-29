# Clinical NLP: Medical Specialty Prediction from Clinical Notes

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()
[![Dataset](https://img.shields.io/badge/Dataset-MTSamples-lightblue)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-92.4%25-success)]()
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
  Step 1:  Load Data & Exploratory Analysis
           ↓
  Step 2:  Filter & Prepare (5 clinically distinct specialties)
           ↓
  Step 3:  Text Preprocessing (spaCy)
           ↓
  Step 4:  Train / Test Split (80/20, stratified)
           ↓
  Step 5:  Baseline — TF-IDF + Logistic Regression
           ↓
  Step 6:  Improved — TF-IDF + Linear SVC
           ↓
  Step 7D: Diagnostic Analysis
           ↓
  Step 7A–C: Evaluation — Confusion Matrix, Model Comparison, Top Terms
           ↓
  Predicted Medical Specialty
```

---

## Why Each Step Matters

| Step | What We Do | Why It Matters |
|---|---|---|
| **📊 1. EDA** | Load data, chart specialty distribution and note lengths, generate word clouds | Reveals class imbalance, note quality, and dominant vocabulary before any modeling |
| **🔧 2. Filter & Prepare** | Keep 5 clinically distinct specialties, drop near-empty notes | Reduces vocabulary overlap between classes — cleaner boundaries = stronger signal |
| **🧹 3. Preprocessing** | Lowercase → remove numbers → strip punctuation → tokenize → remove stopwords → lemmatize | Only truly generic words removed — specialty-specific clinical terms are preserved |
| **✂️ 4. Train/Test Split** | 80/20 stratified split | Stratification ensures every specialty is proportionally represented in both sets |
| **🤖 5. Baseline Model** | TF-IDF + Logistic Regression (20k features, trigrams, C=5) | Sets the performance floor — every subsequent model must beat this |
| **🚀 6. Improved Model** | TF-IDF + Linear SVC (tuned C=0.5) | Sharper decision boundaries in high-dimensional TF-IDF space |
| **📉 7A–C. Evaluation** | Model comparison with 5-fold CV, confusion matrix, top terms per specialty | CV gives honest scores; term analysis validates clinical interpretability |
| **🔬 7D. Diagnostic** | Per-class F1 breakdown, class distribution, note length analysis | Documents the evidence behind every data decision — standard real-world practice |

---

## Version History

| Version | Key Change | Weighted F1 |
|---|---|---|
| v1 | Initial pipeline — top 10 specialties, basic TF-IDF | 0.40 |
| v2 | 6 hand-picked specialties, tuned TF-IDF, cross-validation | 0.52 |
| v3 | Removed Surgery (catch-all class), 5 specialties | **0.924** |

---

## Why Surgery Was Removed

Diagnostic analysis on v2 results revealed that MTSamples "Surgery" is a catch-all label covering cardiac surgery, knee surgery, bowel surgery, and spinal procedures — making its vocabulary overlap with every other specialty. Key findings:

- Surgery made up **45% of the test set** but achieved only **F1: 0.446**
- 133 out of 216 Surgery test notes were misclassified into other specialties
- Removing it is a principled data science decision — Surgery requires a **dedicated sub-classifier** that understands surgical context, planned for Phase 2

---

## Dataset

**MTSamples** — 4,999 real de-identified medical transcription notes across 40 specialties.
- Source: [mtsamples.com](https://www.mtsamples.com/)
- GitHub mirror: [socd06/medical-nlp](https://github.com/socd06/medical-nlp)

**Specialties used (v3):** Cardiovascular/Pulmonary · Orthopedic · Neurology · Gastroenterology · Urology

> ⚠️ No patient data is stored in this repository. The notebook downloads the dataset automatically at runtime.

---

## Results

| Model | Test Accuracy | Weighted F1 | 5-Fold CV F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression | 92.0% | 0.9198 | 0.9033 ± 0.024 |
| **TF-IDF + Linear SVC** | **92.4%** | **0.9236** | **0.9060 ± 0.024** |

**Linear SVC is the best model.** The low CV standard deviation (0.024) confirms the model is stable and generalizes well across different data splits — not a lucky result on one test set.

---

## Key Findings

- **Specialty-specific vocabulary is highly predictive** — top TF-IDF terms align strongly with clinical knowledge (e.g., "arthroscopy", "meniscus" → Orthopedic; "catheterization", "stent" → Cardiovascular; "seizure", "eeg" → Neurology)
- **Trigrams significantly boost performance** — clinical phrases like "anterior cruciate ligament" carry strong specialty signal that unigrams miss
- **Class composition matters more than class count** — Surgery's ambiguous catch-all nature was more damaging than having only 5 classes
- **Cross-validation is essential** — a single train/test split can be misleading; CV F1 of 0.906 confirms the model's reliability

---

## Repository Structure

```
clinical-nlp-specialty-prediction/
│
├── notebooks/
│   ├── clinical_nlp_pipeline.ipynb        ← Main notebook (v1 → v3, all changes inline)
│   └── clinical_nlp_pipeline_v3.ipynb     ← Standalone v3 clean version
│
├── src/
│   ├── preprocess.py                       ← Text cleaning utilities
│   ├── train.py                            ← Model pipeline builders + evaluation
│   └── visualize.py                        ← All chart functions
│
├── data/
│   └── .gitkeep                            ← Place data files here (never committed)
│
├── outputs/
│   └── .gitkeep                            ← Charts saved here at runtime
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

## Project Roadmap

### ✅ Phase 1 — Complete
- MTSamples dataset (4,999 notes, 5 specialties after diagnostic filtering)
- TF-IDF + Logistic Regression baseline — 92.0% accuracy
- TF-IDF + Linear SVC best model — **92.4% accuracy, 0.924 weighted F1**
- Full EDA, confusion matrix, 5-fold CV, top terms per specialty
- Documented diagnostic analysis for every data decision

### 🔄 Phase 2 — In Progress
- MIMIC-III discharge summaries (50,000+ real hospital notes) — PhysioNet credentialing approved ✅
- ICD-9 multi-label code prediction
- Surgery sub-classifier using granular surgical ICD-9 codes
- Bio_ClinicalBERT fine-tuning (`emilyalsentzer/Bio_ClinicalBERT`)
- SHAP explainability — word-level attribution per prediction
- Streamlit app — paste any clinical note, get predicted specialty

---

## Author

**Aruna Kunche**
- Medical Graduate | M.S. Analytics, Harrisburg University (Expected May 2027)
- [LinkedIn](https://www.linkedin.com/in/arunakunche/)
- [GitHub](https://github.com/stethosyntax)

---

## License

MIT License — free to use with attribution.
