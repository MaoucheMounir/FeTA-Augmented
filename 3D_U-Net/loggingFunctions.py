import os
from datetime import datetime
from torch import save
from variables import config

class MultiStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

def get_timestamp():
    now = datetime.now()
    # Formater la date et l'heure pour qu'elles soient compatibles avec un nom de fichier
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Utiliser le timestamp pour nommer le fichier
    return timestamp

def name_log_file(timestamp):
    return os.path.join("logs", f"train_log{timestamp}.log")

def save_model_weights(model_state_dict, timestamp):
        # Save model weights
        weights_path = os.path.join(config['weights_path'], f"model_weights_{timestamp}.pth")#{timestamp}
        save(model_state_dict, weights_path)
