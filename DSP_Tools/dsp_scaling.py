"""
This file contains two sets of functions: Time Stretching and Pitch Scaling.

It supports only mono audio inputs, but multichannel tracks can be condensed
to one channel before a function call.

Three options available:
	1) Pass in pre-read audio array - For inline work
	2) Filename + folder - For a single file
	3) Folder - For applying a stretch function to every file in folder

Pitch Scaling can also be easily combined with bandpass filtering as detailed
function description below.
"""
import os
import librosa
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt

def __validate_factor(factor):
	"""
		Helper function that validates positive factor for time stretching
	"""
	if factor <= 0:
		print('Factor must be greater than 0')
		return False	
	return True

def __validate_filepath(filename, folder):
	"""
		Validates that path exists and file has .wav extension
	"""
	filepath = os.path.join(os.getcwd(), folder, filename)
	if not os.path.exists(filepath):
		print('File ' + filename + ' does not exist')
		return False

	if not filename.lower().endswith('.wav'):
		print('Input file must be a .wav file')
		return False
	
	return True

def __format_array(array):
	"""
		Reformat array as audio sample
	"""
	np.around(array, out=array)
	return array.astype(np.int16)

def plot_specgrams(data, alteredData, Fs):
	"""
		Plots the spectrograms of two audio samples side by side.
		Generally used as before and after applying some scaling

		Args:
			       data : first audio sample
			alteredData : second audio sample
			         Fs : sampling rate
	"""
	figsize = (10, 4)
	NFFT = 1024
	fig, ax = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
	ax[0].specgram(data, NFFT=NFFT, Fs=Fs, noverlap=900)
	ax[0].set_title('Original Audio')
	ax[0].set_xlabel('Time [s]')
	ax[0].set_ylabel('Freq [Hz]')
	ax[0].set_ylim(0, 40000)

	ax[1].specgram(alteredData, NFFT=NFFT, Fs=Fs, noverlap=900)
	ax[1].set_title('Scaled Audio')
	ax[1].set_xlabel('Time [s]')
	ax[1].set_ylabel('Freq [Hz]')
	ax[1].set_ylim(0, 40000)
	plt.show()

def time_stretch(factor, data):
	"""
		Changes the duration of an audio sample without changing the pitch.
		Utilizes the Librosa library to achieve this goal

		Args:
			factor : (must be >0) factor to apply to clip duration
			  data : audio sample to stretch/compress

		Returns:
			stretchedAudio : time stretched audio sample
	"""
	if not __validate_factor(factor):
		return
	if (len(data.shape) == 2):
		data = data.sum(axis=1) / 2
	stretchedAudio = librosa.effects.time_stretch(data.astype(float), factor)
	stretchedAudio = __format_array(stretchedAudio)
	return stretchedAudio

def time_stretch_file(factor: float, filename: str, folder: str='', out: str='output.WAV', write: bool=False):
	"""
		Changes the duration of specified file without changing pitch.
		Delegates to time_stretch function.

		Args:
			  factor : (must be >0) factor to apply to clip duration
			filename : name of file to scale
			  folder : folder where 'filename' is located             --optional
			     out : filename of where to save product              --optional
			   write : True to save product as file, False otherwise  --optional

		Returns:
		                    Fs : sampling rate
			stretchedAudio : time stretched audio sample
	"""
	if not __validate_filepath(filename, folder):
		return	
	filepath = os.path.join(os.getcwd(), folder, filename)
	Fs, data = wav.read(filepath)
	stretchedAudio = time_stretch(factor, data)
	if write:
		wav.write(out, Fs, stretchedAudio)
	return (Fs, stretchedAudio)

def time_stretch_folder(factor: float, inDir: str, outDir: str='stretchedDir'):
	"""
		Changes the duration of each file within the specified folder
		without changing pitch. Delegates to time_stretch function.

		Args:
			  factor : (must be >0) factor to apply to clip duration
			   inDir : folder to run through
			  outDir : folder to save scaling products        --optional
	"""
	inPath = os.path.join(os.getcwd(), inDir)
	if not os.path.exists(inPath):
		print('Input folder does not exist')
		return
	outPath = os.path.join(os.getcwd(), outDir)
	if not os.path.exists(outPath):
		os.mkdir(outDir)
	for filename in os.listdir(inDir):
		if not filename.lower().endswith('.wav'):
			continue
		Fs, stretchedAudio = time_stretch_file(factor, filename, inDir)
		path = os.path.join(os.getcwd(), outDir, filename)
		wav.write(path, Fs, stretchedAudio)

def pitch_shift(targetFreq: float, origFreq: float, Fs, data):
	"""
		Shifts origFreq in the audio sample to targetFreq, preserving
		harmonic relationships. Utilizes Librosa library to achieve this

		Args:
			targetFreq : where origFreq should end up
			  origFreq : frequency of interest to shift to targetFreq
			        Fs : sampling rate of 'data'
			      data : audio sample to shift

		Returns:
			shiftedAudio : pitch shifted audio sample
	"""
	if (len(data.shape) == 2):
		data = data.sum(axis=1) / 2
	semitones = 12 * np.log2( targetFreq / origFreq )
	shiftedAudio = librosa.effects.pitch_shift(data.astype(float), Fs, semitones)
	shiftedAudio = __format_array(shiftedAudio)
	return shiftedAudio

def pitch_shift_file(targetFreq: float, origFreq: float, filename: str, folder: str='', out: str='output.WAV', write: bool=False):
	"""
		Shifts origFreq of specified file to targetFreq. Delegates to pitch_shift function

		Args:
		        targetFreq : where origFreq should end up
			  origFreq : frequency of interest to shift to targetFreq
			  filename : name of file to scale
			    folder : folder where 'filename' is located             --optional
			       out : filename of where to save product              --optional
			     write : True to save product as file, False otherwise  --optional

		Returns:
		                  Fs : sampling rate
			shiftedAudio : pitch shifted audio sample
	"""
	if not __validate_filepath(filename, folder):
		return
	filepath = os.path.join(os.getcwd(), folder, filename)
	Fs, data = wav.read(filepath)
	shiftedAudio = pitch_shift(targetFreq, origFreq, Fs, data)
	if write:
		wav.write(out, Fs, shiftedAudio)
	return (Fs, shiftedAudio)

def pitch_shift_folder(targetFreq: float, origFreq: float, inDir: str, outDir: str='shiftedDir'):
	"""
		Shifts origFreq of all files in inDir to targetFreq.
	       	Delegates to pitch_shift_file function

		Args:
			targetFreq : where origFreq should end up
			  origFreq : frequency of interest to shift to targetFreq
			     inDir : folder to run through
			    outDir : folder to save scaling products       --optional
	"""
	inPath = os.path.join(os.getcwd(), inDir)
	if not os.path.exists(inPath):
		print('Input folder does not exist')
		return
	outPath = os.path.join(os.getcwd(), outDir)
	if not os.path.exists(outPath):
		os.mkdir(outDir)
	for filename in os.listdir(inDir):		
		if not filename.lower().endswith('.wav'):
			continue
		Fs, shiftedAudio = pitch_shift_file(targetFreq, origFreq, filename, inDir)
		path = os.path.join(os.getcwd(), outDir, filename)
		wav.write(path, Fs, shiftedAudio)

def pitch_shift_freq_range(low, high, Fs, data):
	"""
		Applies a bandpass filter from 'low' to 'high'
		and shifts it into the audible range

		Args:
			 low : lower bound of bandpass filter
			high : upper bound of bandpass filter
			Fs   : sampling rate of data
			data : audio sample to isolate and shift

		Returns:
			pitch shifted audio segment of 'low' to 'high'
	"""
	nyq = Fs / 2.0
	filter = sig.butter(2, [low / nyq, high / nyq], btype='bandpass', output='sos')
	filteredSig = sig.sosfiltfilt(filter, data)
	filteredSig = __format_array(filteredSig)
	return pitch_shift(4000, low, Fs, filteredSig)
