from os import fsync
import librosa
from scipy.io import wavfile
import numpy as np
from sklearn import preprocessing

np.set_printoptions(threshold=np.inf)
audio_file = './TEST/ScreamingPiha2.wav'
fs, wavdata = wavfile.read(audio_file)
# wavdata = wavdata / np.linalg.norm(wavdata)
print(wavdata[5000:5100])

print("========================================================================================")

audio, fs = librosa.load(audio_file, sr = fs)
# audio = audio / np.linalg.norm(audio)
print(audio[5000:5100])

print("========================================================================================")

print((wavdata / audio)[5000:5100])

##### SCIPY = LIBROSA * 32768