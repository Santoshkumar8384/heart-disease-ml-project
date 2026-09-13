"""Week 3: model implementation and code documentation."""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .common import load_data, split_data, RESULTS, RANDOM_STATE
from .week2_preprocessing import build_preprocessor

def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE
            ))
        ])
    }

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    for name, model in make_models().items():
        model.fit(X_train, y_train)
        safe_name = name.lower().replace(" ", "_")
        import joblib
        joblib.dump(model, RESULTS / f"{safe_name}_baseline.joblib")
        print(f"{name}: fitted successfully; artifacts saved.")

if __name__ == "__main__":
    main()
