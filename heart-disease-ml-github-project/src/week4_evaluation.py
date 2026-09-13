"""Week 4: model evaluation and validation."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from .common import load_data, split_data, RESULTS, RANDOM_STATE
from .week3_model import make_models

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    fitted = {}
    for name, model in make_models().items():
        cv_scores = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        rows.append({
            "model": name,
            "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
            "cv_precision_mean": cv_scores["test_precision"].mean(),
            "cv_recall_mean": cv_scores["test_recall"].mean(),
            "cv_f1_mean": cv_scores["test_f1"].mean(),
            "cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, prob)
        })
        fitted[name] = (model, pred)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "baseline_metrics.csv", index=False)

    # Confusion matrix for the first baseline.
    name = list(fitted)[0]
    model, pred = fitted[name]
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, pred),
        display_labels=["No disease", "Disease"]
    ).plot()
    plt.title(f"Confusion Matrix – {name}")
    plt.tight_layout()
    plt.savefig(RESULTS / "confusion_matrix.png", dpi=180)
    plt.close()

    metrics.set_index("model")[["test_accuracy","test_precision","test_recall","test_f1","test_roc_auc"]].plot(
        kind="bar", figsize=(10, 6)
    )
    plt.ylabel("Score")
    plt.title("Baseline Model Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison.png", dpi=180)
    plt.close()

    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
