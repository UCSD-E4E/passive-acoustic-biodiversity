from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plot
import glob

#def split_into_n_seconds(wav_data, samplerate, n=10):
#    length_in_seconds = len(wav_data) / samplerate
#    length_in_minutes = length_in_seconds / 60
#    length_in_minutes = int(length_in_minutes)
#    shorter_len = int(length_in_minutes / (1/(60/n)))
#    
#    second_clips = None
#    
#    try: 
#        second_clips = np.split(wav_data, shorter_len)
#        
#    except:
#        cut_wav_data = wav_data[:-((len(wav_data)) % shorter_len)]
#        second_clips = np.split(cut_wav_data, shorter_len)
#
#    print('%d %d-second clips' % (len(second_clips), n))
#    return second_clips

for file in glob.glob("./AM4_samples/" + "*.WAV"):
    print(file)
    pathList = file.split("/")
    clip_name = pathList[len(pathList)-1]
    SAMPLE_RATE,SIGNAL = wavfile.read(file)
    split_clip_sample_size = SAMPLE_RATE * 3
    num_subclips = round(len(SIGNAL)/split_clip_sample_size,0)
    cur_sample = 0
    for ndx in range(int(num_subclips)):
        wavfile.write("./AM4_samples_split/"+clip_name.split(".")[0]+'_'+str(ndx)+".WAV",SAMPLE_RATE,SIGNAL[cur_sample:cur_sample+split_clip_sample_size])
        cur_sample += split_clip_sample_size
