from src.preprocessing import (
    load_dataset,
    load_images,
    split_dataset,
    create_dataset,
)

from src.model_training import (
    build_model,
    train_transfer_learning,
    train_fine_tuning,
)

from src.evaluation import (
    evaluate_model,
    get_classification_report,
)


def main():
    data_path = load_dataset()

    image_paths, labels = load_images(data_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(image_paths, labels)

    train_dataset = create_dataset(X_train, y_train, training=True)

    val_dataset = create_dataset(X_val, y_val)
    test_dataset = create_dataset(X_test, y_test)

    model, base_model = build_model()

    model, history_tl = train_transfer_learning(
        model,
        train_dataset,
        val_dataset,
    )

    model, history_ft = train_fine_tuning(
        model,
        base_model,
        train_dataset,
        val_dataset,
    )

    evaluate_model(model, test_dataset)
    get_classification_report(model, test_dataset)


if __name__ == "__main__":
    main()
