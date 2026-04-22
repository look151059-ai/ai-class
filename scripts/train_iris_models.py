from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def main() -> None:
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_dir = Path(__file__).resolve().parents[1] / "models"

    scaler = StandardScaler().fit(X_train)
    dump(scaler, models_dir / "scaler.joblib")

    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=200).fit(Xs_train, y_train)
    dump(lr, models_dir / "logistic_regression.joblib")

    knn = KNeighborsClassifier(n_neighbors=5).fit(Xs_train, y_train)
    dump(knn, models_dir / "knn.joblib")

    gnb = GaussianNB().fit(X_train.to_numpy(), y_train.to_numpy())
    dump(gnb, models_dir / "gaussian_naive_bayes.joblib")

    # Optional: XGBoost (if installed)
    try:
        from xgboost import XGBClassifier  # type: ignore
    except Exception:
        print("xgboost not installed; skip xgboost model")
        return

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )
    xgb.fit(X_train.to_numpy(), y_train.to_numpy())
    dump(xgb, models_dir / "xgboost.joblib")


if __name__ == "__main__":
    main()

