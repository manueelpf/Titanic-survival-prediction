import os
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    train_path = os.path.join("data", "train.csv")
    test_path = os.path.join("data", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def build_pipeline():
    # Features típicas para un baseline decente
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    target = "Survived"

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(max_iter=2000)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return clf, features, target


def main():
    train_df, test_df = load_data()

    clf, features, target = build_pipeline()

    X = train_df[features]
    y = train_df[target]

    # Cross-validation para estimar rendimiento (accuracy)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: mean={scores.mean():.4f}, std={scores.std():.4f}")

    # Entrena en todo el train y genera predicciones para test
    clf.fit(X, y)

    test_X = test_df[features]
    test_pred = clf.predict(test_X)

    submission = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": test_pred.astype(int)}
    )

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()