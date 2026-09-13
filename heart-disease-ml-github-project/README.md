# Heart Disease Risk Classification using Machine Learning

A six-week Machine Learning Engineer internship project for **YuvaIntern / NSDC**.

## Project objective
Build a reproducible machine-learning pipeline that classifies whether a patient record indicates the presence of heart disease using the public **UCI Heart Disease** dataset.

> **Important:** This is an educational machine-learning project. It is not a medical diagnostic system and must not be used for clinical decisions.

## Six internship tasks

| Week | Task | Implementation |
|---|---|---|
| 1 | Project Planning & Strategy | Pipeline architecture, timeline, resources, risks |
| 2 | Data Preprocessing & Feature Engineering | Missing values, encoding, scaling, validation |
| 3 | Model Implementation & Code Documentation | Logistic Regression and Random Forest |
| 4 | Model Evaluation & Validation | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, Stratified K-Fold |
| 5 | Model Optimization & Experimentation | GridSearchCV and controlled experiments |
| 6 | Final Comprehensive Analysis | End-to-end integration, conclusions and future work |

## Repository structure

```text
heart-disease-ml-project/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── src/
│   ├── common.py
│   ├── week1_planning.py
│   ├── week2_preprocessing.py
│   ├── week3_model.py
│   ├── week4_evaluation.py
│   ├── week5_optimization.py
│   └── week6_final_analysis.py
├── notebooks/
│   └── README.md
├── results/
│   └── README.md
└── reports/
    ├── Week_1_Project_Planning.docx
    ├── Week_2_Data_Preprocessing_and_Feature_Engineering.docx
    ├── Week_3_Model_Implementation_and_Code_Documentation.docx
    ├── Week_4_Model_Evaluation_and_Validation.docx
    ├── Week_5_Model_Optimization_and_Experimentation.docx
    └── Week_6_Final_Comprehensive_ML_Report.docx
```

## Setup

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the complete project:

```bash
python -m src.week6_final_analysis
```

Individual weeks can also be run:

```bash
python -m src.week1_planning
python -m src.week2_preprocessing
python -m src.week3_model
python -m src.week4_evaluation
python -m src.week5_optimization
```

The scripts download the public UCI dataset through `ucimlrepo`, perform the transformations, train models, evaluate them, optimize the selected model and save generated results under `results/`.

## Dataset

UCI Machine Learning Repository – Heart Disease dataset, Dataset ID 45.

The code downloads the dataset at runtime rather than storing a third-party copy in this repository.

## Reproducibility

- Random seed: `42`
- Stratified train/test split
- Stratified 5-fold cross-validation
- Preprocessing is inside scikit-learn Pipelines
- Experiment configuration is recorded in the output CSV/JSON files

## Ethical and technical limitations

The dataset is relatively small and historical. Performance estimates can have substantial variance. A real healthcare system would require larger representative datasets, external validation, calibration, security, governance, clinical review and monitoring.

## Author
Student Machine Learning Engineer Internship Project – YuvaIntern / NSDC
