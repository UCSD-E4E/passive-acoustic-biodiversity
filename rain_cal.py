from __future__ import division
import sys
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from scipy import signal
from scipy.io import wavfile

"""
Detect Rain
Adopted from Automatic Identification of Rainfall in Acoustic Recordings by 
Carol Bedoya et al.

:param data_path: path to recordings
:param thresh_psd: threshold for the mean value of the PSD
:param thresh_snr: Threshold for the signal to noise ratio

:return: mean and std of wind, a binary indicator of false negative
"""
def rain_cal(data_path, t_mean, t_snr):
	sample_rate = None
	recording = []
	for file in glob.glob(data_path + '*.wav'):
		# load wav data
		try:
			rate, data = wavfile.read(file)
		except Exception as e:
			print('(failed) ' + file)
			print('\t' + str(e))
			continue

		recording = np.asarray(data)
		sample_rate = rate
		# import pdb; pdb.set_trace()
		length = recording.shape[0] / sample_rate
		# print(file)
		# print('sample rate = %d' % sample_rate)
		# print('length = %.1fs' % length)

		# import pdb;pdb.set_trace()
		# Stereo to mono
		if recording.ndim == 2:
			recording = recording.sum(axis=1) / 2

		# Downsample to 44.1 kHz
		if sample_rate != 44100:
			recording = signal.decimate(recording, int(sample_rate/44100))
			sample_rate = 44100

		# STEP 1: Estimate PSD vector from signal vector
		f, p = signal.welch(recording, fs=sample_rate, window='hamming', 
							nperseg=512, detrend=False)
		p = np.log10(p)

		# STEP 2: Extract vector a (freq band where rain lies) from PSD vector
		# min and max freq of rainfall freq band
		# divide by sample_rate to normalize from 0 to 1
		rain_min = (2.0 * 600) / sample_rate
		rain_max = (2.0 * 1200) / sample_rate

		limite_inf = int(round(p.__len__() * rain_min))
		limite_sup = int(round(p.__len__() * rain_max))

		# section of interest of the power spectral density
		a = p[limite_inf:limite_sup]
		# print(limite_inf)
		# print(limite_sup)
		# print(a)

		# STEP 3: Compute c (SNR of the PSD in rain freq band)
		# upper part of algorithm 2.1
		mean_a = np.mean(a)
		# lower part of algorithm 2.1
		std_a = np.std(a)

		# snr
		c = mean_a / std_a

		# STEP 4: Classify samples
		if mean_a > t_mean and c > t_snr:
			print(file)
			print('Rainfall of intensity {:.2f}'.format(mean_a))


if __name__ == '__main__':
	### Set your own file path ###
	data_path = './AM15-16-Rain_60s/'
	t_mean = 1e-6
	t_snr = 3.5

	start_time = time.time()
	rain_cal(data_path, t_mean, t_snr)
	print('----- {:.2f} seconds -----'.format(time.time()-start_time))