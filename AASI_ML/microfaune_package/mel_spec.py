import sys
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os

audio_name = sys.argv[1]
y, sr = librosa.load(audio_name)


spec = np.abs(librosa.stft(y,hop_length=512))
spec = librosa.amplitude_to_db(spec, ref = np.max)
librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('spectrogram')
plt.show()
#plt.savefig(directory_str + '/'+ audio_name[:-4] + "_spec")
plt.close()

