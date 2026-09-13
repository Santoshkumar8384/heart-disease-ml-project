"""Week 1: project planning and strategy."""
from .common import save_json

def main():
    plan = {
        "project": "Heart Disease Risk Classification using Machine Learning",
        "problem": "Binary classification of heart-disease presence from structured clinical features.",
        "timeline": {
            "week_1": "Planning and strategy",
            "week_2": "Preprocessing and feature engineering",
            "week_3": "Model implementation",
            "week_4": "Evaluation and validation",
            "week_5": "Optimization and experimentation",
            "week_6": "Final integration and analysis"
        },
        "critical_path": [
            "data validation", "leakage-safe preprocessing",
            "baseline model", "validation", "optimization", "final evaluation"
        ],
        "resources": ["Python", "pandas", "NumPy", "scikit-learn", "matplotlib", "seaborn", "Git/GitHub"],
        "major_risks": [
            "data leakage", "overfitting", "missing values",
            "small sample size", "class imbalance", "reproducibility"
        ]
    }
    save_json(plan, "week1_project_plan.json")
    print("Week 1 plan saved to results/week1_project_plan.json")

if __name__ == "__main__":
    main()
