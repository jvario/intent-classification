"""The classifier interface. You can modify/refactor it according to your needs."""
import pickle

from config import MODEL_FILE_PATH, TRAIN_DATA_FILE_PATH
from utiils import read_data

class IntentClassifier:

    def    __init__(self) -> None:
        pass

    def is_ready(self) -> bool:

        return True

    def load(self, file_path) -> None:
        """Load the model or configuration the specified file path."""
        pass

