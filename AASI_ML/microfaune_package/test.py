from microfaune.detection import RNNDetector
import matplotlib.pyplot as plt
import sys
import numpy as np

audio_name = sys.argv[1]
detector = RNNDetector()
global_score, local_score = detector.predict_on_wav(audio_name)

print(global_score)

#t = np.arange(0,len(local_score),1)
t = np.arange(0,10.0,10/len(local_score))
print(len(local_score))
print(len(t))
plt.plot(t,local_score,lw=1);
plt.ylim(0,1.0)
plt.show()

