import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70

def _configure_mlflow() -> None:
    """
    Cau hinh MLflow an toan cho CI/test:
    - Khong phu thuoc vao mlflow.db/mlruns co san trong repo.
    - Tranh artifact_uri tuyet doi tu may local (vd: /home/duckduck/...).
    """
    is_ci = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
    is_pytest = "PYTEST_CURRENT_TEST" in os.environ

    if is_ci or is_pytest:
        tracking_dir = os.path.join(tempfile.gettempdir(), "mlflow_vinuni")
        os.makedirs(tracking_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file://{tracking_dir}")

    mlflow.set_experiment("vinuni")
    exp = mlflow.get_experiment_by_name("vinuni")
    print(
        "[MLFLOW DEBUG] "
        f"is_ci={is_ci} is_pytest={is_pytest} "
        f"tracking_uri={mlflow.get_tracking_uri()} "
        f"experiment_id={exp.experiment_id if exp else 'None'} "
        f"artifact_location={exp.artifact_location if exp else 'None'}"
    )


_configure_mlflow()
def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    with mlflow.start_run():
        active_run = mlflow.active_run()
        print(
            "[MLFLOW DEBUG] "
            f"run_id={active_run.info.run_id if active_run else 'None'} "
            f"artifact_uri={mlflow.get_artifact_uri()}"
        )
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
