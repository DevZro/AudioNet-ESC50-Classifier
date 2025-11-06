import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def precompute_spectrograms(start_path, end_path, group, dpi=50):
    files = Path(start_path).glob('*.wav')
    for filename in files:
        if filename.name[0] in group:
            audio_tensor, sr = librosa.load(filename, sr=None)
            spectrogram = librosa.feature.melspectrogram(y=audio_tensor, sr=sr)
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            librosa.display.specshow(log_spectrogram, sr=sr, x_axis=None, y_axis=None)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.gcf().savefig(end_path + "/" + filename.name[:-3] + "png", dpi=dpi)

endpoints = ["train", "valid", "test"]
groups = [["1", "2", "3"], ["4"], ["5"]]

for endpoint, group in zip(endpoints, groups):
    precompute_spectrograms("ESC-50-master/audio", "data/" + endpoint, group)
    print(f"{endpoint} Done!")