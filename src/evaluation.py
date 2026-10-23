import numpy as np

import mlflow
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.config import CLASS_NAMES


def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset)

    mlflow.log_metrics(
        {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        }
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy


def get_classification_report(model, test_dataset):
    y_true = np.concatenate([labels.numpy() for _, labels in test_dataset])

    y_probabilities = model.predict(test_dataset)

    y_pred = np.argmax(y_probabilities, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    matrix = confusion_matrix(y_true, y_pred)

    mlflow.log_text(report, "evaluation/classification_report.txt")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    output_path = Path("confusion_matrix.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(output_path))
    output_path.unlink(missing_ok=True)

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    return report, matrix
