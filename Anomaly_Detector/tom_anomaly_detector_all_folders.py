from __future__ import division
import numpy as np
import glob
import time
import csv
from scipy import signal
from scipy.io import wavfile
import os


def run_script_on_dir(pathToDir, t_mean, t_snr):
    #Step 1: get all folders in directory
    folders = get_all_folders_in_directory(pathToDir)

    #Step 2: run rain cal on all files in every folder
    for folder in folders:
        #print(folder)
        data_path = folder + os.path.sep
        rain_cal(data_path, t_mean, t_snr)


def get_all_folders_in_directory(pathToDir):
    folders = []
    for x in os.walk(pathToDir):
        folders.append(x[0])
    folders.pop(0)
    return folders


def rain_cal(data_path, t_mean, t_snr):
    print("In path: ", data_path)
    file_detected_count  = 0
    file_evaluated_count = 0
    corrupted_file       = 0
    downsample_issue     = 0
    for file in glob.glob(data_path + '*.WAV'):
        file_detected_count += 1 
        # load wav data
        try:
            # Create a new row
            # Write the clip name
            # Write the folder name
            #print("filePath:  ", file)
            rate, data = wavfile.read(file)
        except Exception as e:
            print('failed to find file @ path:' + file)
            print('\t' + str(e))
            continue

        if len(data) == 0:
            #print("Corrupted file, no data")
            corrupted_file += 1
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
        recording, sample_rate, success = downsample(file, recording, sample_rate, hertz=44100) # TODO:: Tom saftey
        if not success:
            downsample_issue += 1
            continue
	#if sample_rate != 44100:
	#    recording = signal.decimate(recording, int(sample_rate/44100))
	#    sample_rate = 44100

        # STEP 1: Estimate PSD vector from signal vector
        f, p = signal.welch(recording, fs=sample_rate, window='hamming',
                            nperseg=512, detrend=False)
        p = np.log10(p)

        # STEP 2: Extract vector a (freq band where rain lies) from PSD vector
        # min and max freq of rainfall freq band
        # divide by sample_rate to normalize from 0 to 1
        # write the lower frequency range
        rain_min = (2.0 * 600) / sample_rate
        # Write the upper frequency range
        rain_max = (2.0 * 1200) / sample_rate

        limite_inf = int(round(p.__len__() * rain_min))
        limite_sup = int(round(p.__len__() * rain_max))

        # section of interest of the power spectral density
        a = p[limite_inf:limite_sup]

        # STEP 3: Compute c (SNR of the PSD in rain freq band)
        # upper part of algorithm 2.1
        mean_a = np.mean(a)
        # print(mean_a)
        # lower part of algorithm 2.1
        std_a = np.std(a)

        # snr
        c = mean_a / std_a
        # print(c)
        pathList = file.split(str(os.path.sep))
        
        # Write the signal to noise ratio
        with open('6_Audiomoths.csv', mode='a') as test_file:
	    test_writer = csv.writer(test_file,delimiter=',')
	    test_writer.writerow([pathList[len(pathList)-1],pathList[len(pathList)-2],600,1200,round(c,2),round(mean_a,2)])
            file_evaluated_count += 1
	# STEP 4: Classify samples
	if mean_a > t_mean and c > t_snr:
            print('{}: Noise of intensity {:.2f}'.format(file, mean_a))
    pathList = data_path.split(str(os.path.sep))
    with open('errors.csv', mode='a') as errors:
        test_writer = csv.writer(errors, delimiter=',')
        test_writer.writerow([pathList[len(pathList) - 2], corrupted_file,
            downsample_issue, file_detected_count, file_evaluated_count])

def downsample(fileName, recording, sample_rate, hertz):
    try:
        if sample_rate != hertz:
            recording = signal.decimate(recording, int(sample_rate / hertz))
            sample_rate = hertz
            return recording, sample_rate, True
        else:
            #raise ValueError(" sample_rate was NOT equal to hertz")
            print("sample_rate was NOT equal to hertz")
            return recording, sample_rate, False
    except Exception as e:
        print('Failed to dowmsample file @ path: ' + fileName)
        print('\t' + str(e))
        return recording, sample_rate, False

if __name__ == '__main__':
    ### Set your own file path ###
    #            /media/e4e/Audio UCSD/CARPETA-GRABADORES MADERACRE-OTORONGO-CHULLACHAQUI-2019
    path       = "/media/e4e/Audio UCSD/CARPETA-GRABADORES MADERACRE-OTORONGO-CHULLACHAQUI-2019/"
    print("I am live")

    # Adjusted for trucks
    t_mean = 2
    # t_mean = 1e-6
    t_snr = 3.5
    run_script_on_dir(path, t_mean, t_snr)

    '''
    start_time = time.time()
    rain_cal(data_path, t_mean, t_snr)
    print('----- {:.2f} seconds -----'.format(time.time() - start_time))
    '''

    #data_path = "./GRABADOR-WWF-1/"
    #start_time = time.time()
    #rain_cal(data_path, t_mean, t_snr)
    #print('----- {:.2f} seconds -----'.format(time.time()-start_time))
