import sys
import os
from pydub import AudioSegment
folder_str = sys.argv[1]
tar_str = sys.argv[2]
#folder_str = "/Users/fyy0194/Documents/cse237D/grabador2019_1/output/"

#tar_str = "/Users/fyy0194/Documents/cse237D/grabador2019_1/output/trimed/"

for file in os.listdir(folder_str):
    audio_name = os.fsdecode(file)
    if audio_name.endswith(".WAV"):
        loop = AudioSegment.from_wav(folder_str+'/'+audio_name)
        length = len(loop)
        for i in range(0,(int) (length/10000)):
            newAudio = loop[i*10000 : i*10000+10000]
            newAudio.export(tar_str+audio_name[:-4]+'_'+str(i)+'.wav',format = "wav")
            
        
