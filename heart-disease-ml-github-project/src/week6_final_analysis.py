"""Week 6: final end-to-end analysis."""
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .common import load_data, split_data, RESULTS
from .week3_model import make_models

def main():
    # Run evaluation and optimization first when needed.
    from .week4_evaluation import main as eval_main
    from .week5_optimization import main as opt_main
    eval_main()
    opt_main()

    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    import joblib
    optimized = joblib.load(RESULTS / "optimized_model.joblib")
    optimized.fit(X_train, y_train)
    pred = optimized.predict(X_test)
    prob = optimized.predict_proba(X_test)[:, 1]

    final = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob)
    }
    (RESULTS / "final_metrics.json").write_text(json.dumps(final, indent=2))

    pd.DataFrame([final]).to_csv(RESULTS / "final_metrics.csv", index=False)

    print("Final held-out test metrics:")
    for k, v in final.items():
        print(f"{k}: {v:.4f}")

    print("\\nImportant: this is an educational prototype, not a medical diagnostic system.")

if __name__ == "__main__":
    main()
