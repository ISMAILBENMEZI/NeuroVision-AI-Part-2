import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
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


def train_transfer_learning(model, train_dataset, val_dataset):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

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

    # model = tf.keras.models.load_model(MODELS_DIR / "transfer_learning_best.keras")

    return model, history


def train_fine_tuning(model, base_model, train_dataset, val_dataset):
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

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FINE_TUNING_EPOCHS,
        callbacks=get_callbacks(MODELS_DIR / "fine_tuning_best.keras"),
    )

    return model, history
