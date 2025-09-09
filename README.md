# CS4680–A1: Student Stress Classification

This project applies classification to predict student academic stress levels (Low, Moderate, High) using survey data.

## Files
- **train_classification.py** – Main script: loads data, preprocesses features, trains Logistic Regression & Random Forest models, evaluates performance, and saves results.
- **academic Stress level - maintainance 1.csv** – The dataset, downloaded from Kaggle (*Student Academic Stress – Real World Dataset*).
- **requirements.txt** – List of Python dependencies needed to run the project.
- **plots/** – Contains confusion matrix images for both models.
- **artifacts/** – Stores predictions (`test_predictions.csv`) for the test set.

## Dataset
- Source: [Kaggle – Student Academic Stress (Real World Dataset)](https://www.kaggle.com/datasets/poushal02/student-academic-stress-real-world-dataset)  
- Description: Survey responses from students, including academic stage, peer/home pressure, study environment, coping strategies, and self-rated stress index (1–5).  
- Target variable: `Rate your academic stress index` mapped into **Low (1–2), Moderate (3), High (4–5)**.

## Setup & Running
1. Clone or download this repository.
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # macOS/Linux
3. pip install -r requirements.txt
4. python train_classification.py

