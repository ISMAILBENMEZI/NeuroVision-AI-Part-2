from pathlib import Path


import kagglehub
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import (
    DATASET_ID,
    DATASET_FOLDER,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_STATE,
    SUPPORTED_EXTENSIONS,
    # TRAIN_SIZE,
    # VALIDATION_SIZE,
    # TEST_SIZE,
)


def load_dataset():
    dataset_path = Path(kagglehub.dataset_download(DATASET_ID))
    data_path = dataset_path / DATASET_FOLDER

    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_path}")
    for class_name in CLASS_NAMES:
        class_path = data_path / class_name
        if not class_path.is_dir():
            raise FileNotFoundError(f"Class folder not found: {class_path}")
    return data_path


def load_images(data_path):
    image_paths = []
    labels = []

    for label, class_name in enumerate(CLASS_NAMES):
        class_path = data_path / class_name

        for image_path in class_path.rglob("*"):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            try:
                image = tf.io.read_file(str(image_path))
                image = tf.io.decode_image(
                    image,
                    channels=3,
                    expand_animations=False,
                )

                image.numpy()

                image_paths.append(str(image_path))
                labels.append(label)
            except (tf.error.OpError, ValueError) as error:
                print(f"Skipping corrupted image: {image_path}")
    if not image_paths:
        raise ValueError("No valid images were found.")
    return np.array(image_paths) , np.array(labels , dtype=np.int32)


def split_dataset(image_paths , labels):

    X_train , X_temp , y_train , y_temp = train_test_split(
        image_paths,
        labels,
        test_size= 0.36,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    X_val , X_test , y_val , y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5556,
        random_state=42,
        stratify=y_temp,
    )

    return X_train , X_val , X_test , y_train , y_val , y_test

def preprocess_image(image_path , label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(
        image,
        channels = 3,
        expand_animations = False,
    )

    image.set_shape([None , None , 3])

    image = tf.image.resize(image , IMAGE_SIZE)
    image = preprocess_input(image)

    return image , label

def create_dataset(image_paths , labels , training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths , labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=RANDOM_STATE,
        )

    dataset = dataset.map(
        preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
