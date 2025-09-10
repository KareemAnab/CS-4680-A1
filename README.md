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

## Code Analysis

### Data Preprocessing
- Dropped unused columns (`Timestamp`, raw numeric stress index, and target itself).
- Split features into **numeric** (peer/home pressure, competition rating) and **categorical** (academic stage, environment, coping strategy, habits).
- Used a `ColumnTransformer`:
  - Numeric: scaled with `StandardScaler`.
  - Categorical: encoded with `OneHotEncoder`.
- Wrapped preprocessing in a `Pipeline` to prevent data leakage.

### Model Training
- Two classifiers were applied:
  - **Logistic Regression** – linear, interpretable baseline.
  - **Random Forest** – non-linear ensemble with stronger generalization.
- Both trained and validated using a 80/20 train/test split.

### Evaluation
- 5-fold stratified cross-validation applied on the training set using **macro F1** (balances class performance).
- Final evaluation metrics (on the test set): **accuracy, precision, recall, F1-score**.
- Confusion matrices saved for each model in the `plots/` folder.

### Best Model Selection
- Models compared by **macro F1**.
- Predictions from the best model are saved into `artifacts/test_predictions.csv`.

## Model Comparison

Both models were evaluated on the same 20% held-out test set and also 5-fold stratified CV was used on the training set (macro F1).

| Model              | Test Accuracy | Test Macro F1 |
|--------------------|---------------|---------------|
| Logistic Regression| 0.643         | 0.536         |
| Random Forest      | 0.714         | 0.699         |

**Discussion for Suitability:**
- **Random Forest** achieved the highest accuracy and macro-F1, indicating better balanced performance across the three classes (Low/Moderate/High), and it captures non-linear interactions between survey features.  
- **Logistic Regression** is a strong linear baseline and more interpretable, but it struggled with the **Moderate** class (class imbalance + non-linear boundaries).  
- **Choice:** For this dataset, **Random Forest** is more suitable if the objective is predictive performance; Logistic Regression remains useful for interpretability/baseline reporting.  

## Findings

- **Logistic Regression**
  - Accuracy ≈ **64%**, Macro F1 ≈ **0.54**.
  - Strong on *Low* and *High* classes but **failed to predict Moderate**.
  - Indicates linear separation is not sufficient for imbalanced data.

- **Random Forest**
  - Accuracy ≈ **71%**, Macro F1 ≈ **0.70**.
  - Strong results on *Low* and *High*, Moderate class weaker but still better than Logistic Regression.
  - Handles non-linear patterns better.

- **Overall Conclusion**
  - Random Forest performed best overall.
  - Both models show difficulty with the underrepresented **Moderate** class.
  - Future improvements: oversampling, class weighting, or more data collection.
