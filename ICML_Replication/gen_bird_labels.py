import sys

sys.path.append("../")
from PyHa.PyHa.statistics import *
from PyHa.PyHa.IsoAutio import *
from PyHa.PyHa.visualizations import *
import pandas as pd
import os

path_to_audio_files = "../Stratified_Random_Sample_Peru2019_Subset/"
path_to_save = "SRS_BirdNET_Labels_wConf.csv"

# generate a list of audiomoths
remove = [18, 19, 21, 28]
am_list = [str(x) for x in range(1,31) if x not in remove]
am_list = ["AM" + x for x in am_list]
am_list.extend(["WWF" + str(x) for x in range(1,6)])

isolation_parameters = {
   "model" : "birdnet",
   "output_path" : "outputs",
   "filetype" : "wav", 
   "num_predictions" : 1,
   "write_to_csv" : True
}


for am in am_list:
    print(am)
    folder = path_to_audio_files + am + "_samples/"
    automated_df = generate_automated_labels(folder, isolation_parameters)
    automated_df.to_csv(am +  "_" + path_to_save)
