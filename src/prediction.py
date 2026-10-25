import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import CLASS_NAMES, IMAGE_SIZE, MODELS_DIR


MODEL_PATH = MODELS_DIR / "fine_tuning_best.keras"


def load_prediction_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image_file):
    image_bytes = image_file.getvalue()

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = preprocess_input(image)

    image = tf.expand_dims(image, axis=0)

    return image


def predict_image(model, image_file):
    image = preprocess_image(image_file)

    probabilities = model.predict(image, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(probabilities[predicted_index])

    class_probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities,
        )
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": class_probabilities,
    }
