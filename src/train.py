import argparse
import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Step 1: load data and split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target,
    test_size=args.test_size,
    random_state=args.random_state,
    stratify=iris.target,
)

mlflow.set_experiment("iris-classification")

with mlflow.start_run():
    # Step 1: train
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)

    # Step 1: evaluate
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro"),
        "recall_macro": recall_score(y_test, preds, average="macro"),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }

    # Step 2: log parameters, metrics and model to MLflow
    mlflow.log_params(vars(args))
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    # Save a copy for the API and Docker
    joblib.dump(model, "models/model.pkl")

    print("Parameters:", vars(args))
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
