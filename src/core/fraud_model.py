import shutil
import numpy as np
from pathlib import Path
from tensorflow import keras
from huggingface_hub import snapshot_download

from utils import (
    get_logger,
    ai_config
)

logger = get_logger(__name__)

class FraudDetectionModel:
    def __init__(self) -> None:
        self.repo_id = ai_config.FRAUD_MODEL_REPO_ID
        self.model_path = ai_config.FRAUD_MODEL_LOCAL_PATH
        self.model = None
        
        self.load_model()

    def _download_model(self) -> None:
        if Path(self.model_path).exists():
            logger.info(f"Removing existing model from {self.model_path}")
            shutil.rmtree(self.model_path)

        logger.info(f"Downloading model from {self.repo_id}")
        snapshot_download(
            repo_id=self.repo_id,
            repo_type="model",
            local_dir=self.model_path,
            allow_patterns=["saved_model.pb", "variables/*", "assets/*", "keras_metadata.pb"],
        )
        logger.info(f"Model downloaded to {self.model_path} successfully")
    
    def initialize_model(self) -> None:
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(30,)),
                keras.layers.TFSMLayer(
                    self.model_path,
                    call_endpoint="serving_default"
                ),
            ]
        )

    def load_model(self) -> None:
        logger.info(f"Loading model from {self.model_path}")
        try:
            self.initialize_model()
        except:
            logger.warning(f"Failed to load model from {self.model_path}. Trying to download again...")
            self._download_model()
            self.initialize_model()
        finally:
            if self.model is None:
                raise RuntimeError("Model failed to load.")
            logger.info(f"Model loaded from {self.model_path} successfully")

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.model.predict(features, verbose=0)
        list_prob = probabilities["dense_7"][0].tolist()

        binary_result = [1 if prob > ai_config.FRAUD_THRESHOLD else 0 for prob in list_prob]
        return binary_result


if __name__ == "__main__":
    model = FraudDetectionModel()
    sample = np.array([[0.0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, 
-0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 
0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, 
-0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 
0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, 
-0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 
0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]]).astype("float32")
    print(model.predict(sample))