#import the pyplot and wavfile modules 

import matplotlib.pyplot as plot

from scipy.io import wavfile

 

# Read the wav file (mono)

samplingFrequency, signalData = wavfile.read('Bird2.WAV')

 

# Plot the signal read from wav file

# plot.subplot(211)

# plot.title('Spectrogram of a wav file with rainforest sounds')

 

# plot.plot(signalData)

# plot.xlabel('Sample')

# plot.ylabel('Amplitude')

 

# plot.subplot(212)

plot.specgram(signalData,Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')

 

plot.show()
