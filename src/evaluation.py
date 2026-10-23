import numpy as np

from sklearn.metrics import classification_report, confusion_matrix
from src.config import CLASS_NAMES


def evaluate_model(model, test_dataset):
    test_loss, test_accuracy = model.evaluate(test_dataset)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy


def get_classification_report(model, test_dateset):
    y_true = np.concatenate([labels.numpy() for _, labels in test_dateset])

    y_probabilities = model.predict(test_dateset)

    y_pred = np.argmax(y_probabilities, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )

    matrix = confusion_matrix(y_true, y_pred)

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    return report , matrix


