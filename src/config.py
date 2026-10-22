from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_ID = "ismailbenmezi/dataset-neuron-vision"
DATASET_FOLDER = "Data"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = (224 , 224)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32

# TRAIN_SIZE = 0.64
# VALIDATION_SIZE = 0.16
# TEST_SIZE = 0.20
RANDOM_STATE = 42

TRANSFER_LEARNING_EPOCHS = 15
TRANSFER_LEARNING_LR = 1e-3
FINE_TUNE_LAST_LAYERS = 30

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}