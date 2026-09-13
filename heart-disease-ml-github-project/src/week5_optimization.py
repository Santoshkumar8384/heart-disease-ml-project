"""Week 5: hyper-parameter optimization and experimentation."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .common import load_data, split_data, RESULTS, RANDOM_STATE
from .week2_preprocessing import build_preprocessor

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    experiments = []

    lr = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    lr_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"]
    }
    lr_search = GridSearchCV(lr, lr_grid, scoring="f1", cv=cv, n_jobs=-1, return_train_score=True)
    lr_search.fit(X_train, y_train)
    experiments.append({
        "model": "Logistic Regression",
        "best_cv_f1": lr_search.best_score_,
        "best_params": str(lr_search.best_params_)
    })

    rf = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))
    ])
    rf_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    }
    rf_search = GridSearchCV(rf, rf_grid, scoring="f1", cv=cv, n_jobs=-1, return_train_score=True)
    rf_search.fit(X_train, y_train)
    experiments.append({
        "model": "Random Forest",
        "best_cv_f1": rf_search.best_score_,
        "best_params": str(rf_search.best_params_)
    })

    pd.DataFrame(experiments).to_csv(RESULTS / "optimization_results.csv", index=False)

    # Save detailed CV results for reproducibility.
    pd.DataFrame(lr_search.cv_results_).to_csv(RESULTS / "logistic_cv_results.csv", index=False)
    pd.DataFrame(rf_search.cv_results_).to_csv(RESULTS / "random_forest_cv_results.csv", index=False)

    import joblib
    best = rf_search if rf_search.best_score_ >= lr_search.best_score_ else lr_search
    joblib.dump(best.best_estimator_, RESULTS / "optimized_model.joblib")

    print(pd.DataFrame(experiments).to_string(index=False))
    print("Best model saved to results/optimized_model.joblib")

if __name__ == "__main__":
    main()
