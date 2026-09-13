"""Week 2: data preprocessing and feature engineering."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .common import load_data, split_data, NUMERIC_FEATURES, RANDOM_STATE, RESULTS

# Include the discrete 'ca' feature as categorical.
CAT_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
NUM_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BIN_FEATURES = ["sex", "fbs", "exang"]

def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUM_FEATURES),
        ("cat", categorical_pipe, CAT_FEATURES),
        ("bin", binary_pipe, BIN_FEATURES)
    ])

def main():
    X, y = load_data()

    audit = pd.DataFrame({
        "missing_values": X.isna().sum(),
        "missing_percent": X.isna().mean() * 100
    })
    audit["dtype"] = X.dtypes.astype(str)
    audit.to_csv(RESULTS / "data_quality_audit.csv")

    duplicate_count = int(X.duplicated().sum())
    pd.DataFrame({"duplicate_rows": [duplicate_count]}).to_csv(
        RESULTS / "duplicate_audit.csv", index=False
    )

    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    pd.DataFrame({
        "train_rows": [len(X_train)],
        "test_rows": [len(X_test)],
        "transformed_train_columns": [X_train_ready.shape[1]],
        "transformed_test_columns": [X_test_ready.shape[1]],
        "positive_rate_train": [y_train.mean()],
        "positive_rate_test": [y_test.mean()]
    }).to_csv(RESULTS / "preprocessing_validation.csv", index=False)

    print(audit)
    print("Transformed shape:", X_train_ready.shape, X_test_ready.shape)

if __name__ == "__main__":
    main()
