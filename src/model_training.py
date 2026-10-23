import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

import mlflow
import mlflow.tensorflow

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.config import (
    IMAGE_SIZE,
    IMAGE_CHANNELS,
    NUM_CLASSES,
    MODELS_DIR,
    TRANSFER_LEARNING_EPOCHS,
    TRANSFER_LEARNING_LR,
    FINE_TUNING_EPOCHS,
    FINE_TUNING_LR,
    FINE_TUNE_LAST_LAYERS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    BATCH_SIZE,
)


def build_model():
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS)
    )

    base_model.trainable = False

    inputs = layers.Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS))

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
    )(x)

    model = models.Model(inputs, outputs)

    return model, base_model


def get_callbacks(model_path):
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
        ),
    ]


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def train_transfer_learning(model, train_dataset, val_dataset):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    setup_mlflow()
    
    mlflow.tensorflow.autolog(log_models=False)
    with mlflow.start_run(run_name="Transfer-Learning"):
        mlflow.log_params(
            {
                "model_name": "MobileNetV2",
                "training_stage": "transfer_learning",
                "epochs": TRANSFER_LEARNING_EPOCHS,
                "learning_rate": TRANSFER_LEARNING_LR,
                "batch_size": BATCH_SIZE,
                "image_size": str(IMAGE_SIZE),
            }
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRANSFER_LEARNING_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=TRANSFER_LEARNING_EPOCHS,
        callbacks=get_callbacks(MODELS_DIR / "transfer_learning_best.keras"),
    )

    best_val_accuracy = max(history.history["val_accuracy"])
    best_val_loss = min(history.history["val_loss"])

    mlflow.log_metric(
        {
            "best_val_accuracy": float(best_val_accuracy),
            "best_val_loss": float(best_val_loss),
        }
    )

    mlflow.log_artifact(str(MODELS_DIR / "transfer_learning_best.keras"))

    # model = tf.keras.models.load_model(MODELS_DIR / "transfer_learning_best.keras")

    return model, history


def train_fine_tuning(model, base_model, train_dataset, val_dataset):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    base_model.trainable = True

    for layer in base_model.layers[:-FINE_TUNE_LAST_LAYERS]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FINE_TUNING_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    setup_mlflow()
    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(run_name="Fine-Tuning"):
        mlflow.log_params(
            {
                "model_name": "MobileNetV2",
                "training_stage": "fine_tuning",
                "epochs": FINE_TUNING_EPOCHS,
                "learning_rate": FINE_TUNING_LR,
                "fine_tune_last_layers": FINE_TUNE_LAST_LAYERS,
                "batch_size": BATCH_SIZE,
                "image_size": str(IMAGE_SIZE),
            }
        )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FINE_TUNING_EPOCHS,
        callbacks=get_callbacks(MODELS_DIR / "fine_tuning_best.keras"),
    )

    mlflow.log_metrics(
        {
            "best_val_accuracy": float(max(history.history["val_accuracy"])),
            "best_val_loss": float(min(history.history["val_loss"])),
        }
    )

    mlflow.log_artifact(str(MODELS_DIR / "fine_tuning_best.keras"))

    return model, history
