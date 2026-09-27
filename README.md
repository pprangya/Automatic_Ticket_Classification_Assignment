# Automatic Ticket Classification

## Problem Statement

A financial company receives thousands of unstructured customer complaint tickets daily across products such as credit cards, bank accounts, mortgages/loans, and more. Manually routing each ticket to the correct department is slow and error-prone.

This project builds an **automatic ticket classification system** using **Non-Negative Matrix Factorization (NMF)** for unsupervised topic modelling, followed by supervised classification models to predict the category of any new complaint.

---

## Dataset

| File | Description |
|------|-------------|
| `complaints-2021-05-14_08_16.zip` | 78,313 customer complaints with 22 features in JSON format. Unzip before running the notebook. |

**Source:** Consumer Financial Protection Bureau (CFPB) public complaints dataset.

---

## Five Complaint Categories

| Category |
|----------|
| Credit card / Prepaid card |
| Bank account services |
| Theft / Dispute reporting |
| Mortgages / Loans |
| Others |

---

## Pipeline

Data Loading → Load JSON, convert to DataFrame
Text Preprocessing → Clean text, Lemmatize (spaCy), POS-filter (keep nouns only)
EDA → Character length distribution, Word Cloud, N-gram analysis
Feature Extraction → TF-IDF (max_df=0.95, min_df=2)
Topic Modelling → NMF (n_components=5, random_state=40)
Supervised Learning → CountVectorizer + TF-IDF → Train/Test Split
Model Training → Logistic Regression, Decision Tree, Random Forest
Model Inference → Predict category for new complaint text


---

## Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model |
| Decision Tree | Non-linear, interpretable |
| Random Forest | Ensemble, best accuracy |

Evaluation metrics: **Accuracy**, **Classification Report** (Precision / Recall / F1), **Confusion Matrix**

---

## How to Run

1. Unzip the dataset:
unzip complaints-2021-05-14_08_16.zip



2. Install dependencies:
pip install numpy pandas scikit-learn nltk spacy wordcloud matplotlib seaborn plotly
python -m spacy download en_core_web_sm



3. Open and run the notebook top to bottom:
Automatic_Ticket_Classification_Assignment_PrangyaPradhan.ipynb



> **Note:** spaCy lemmatization on ~50,000 rows takes approximately 20–40 minutes depending on hardware.

---

## Notebook

`Automatic_Ticket_Classification_Assignment_PrangyaPradhan.ipynb` — contains all code, outputs, visualisations, and model results.
