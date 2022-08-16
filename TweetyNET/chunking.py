import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import shutil, os
#from PyHa.statistics import *
#from PyHa.IsoAutio import *
#from PyHa.visualizations import *
#from PyHa.annotation_post_processing import *
import pandas as pd
import numpy as np

def extact_split(df, chunk_len=3):
    def gen_list(a, b, chunk):
        # generate multiples of 3 between a and b
        multiplist = [i for i in range(math.ceil(a), math.ceil(b)) if i % chunk == 0]
        # if there are multiples
        if multiplist:
            # insert start and end times
            multiplist.insert(0,a)
            multiplist.append(b)
            # create a list of list ranges
            multiplist = [[multiplist[i], multiplist[i+1]] for i in range(len(multiplist) - 1)]
        return multiplist

    # create an end time column
    df["END_TIME"] = df["OFFSET"] + df["DURATION"]
    df["SPLIT"] = df.apply(lambda x: gen_list(x["OFFSET"], x["END_TIME"], chunk_len), axis=1)
    df_split = df.copy()
    df_split = df_split.explode("SPLIT", ignore_index=True)

    # assign start time to index 0 in "SPLIT" range
    df_split["START_TIME"] = df_split["SPLIT"].dropna().map(lambda x: x[0])
    df_split["START_TIME"] = df_split["START_TIME"].fillna(df_split["OFFSET"])
    # assign end time to index 1 in "SPLIT" range
    df_split["END_TIME"] = df_split["SPLIT"].dropna().map(lambda x: x[1])
    df_split["END_TIME"] = df_split["END_TIME"].fillna(df_split["OFFSET"] + df_split["DURATION"])

    df_split["OFFSET"] = df_split["START_TIME"]
    df_split["DURATION"] = df_split["END_TIME"] - df_split["START_TIME"]
    return df_split

def fill_no_class(df, chunk):
    new_df = df.copy()
    count = 0
    files = np.unique(df["IN FILE"])
    for file in files:
        print(count, len(files))
        count += 1
        
        sub_df = df[df["IN FILE"] == file]
        clip_length = sub_df.iloc[0]["CLIP LENGTH"]
        chunks = list(range(int(clip_length // chunk + 1)))
        #print("===============================================")
        #print(chunks, np.unique(sub_df["CLIP LENGTH"]), clip_length)
        chunks_with_annotations = np.array(sub_df["CHUNK_ID"].apply(int))
        #print(chunks_with_annotations)
        no_class_chunks = np.setdiff1d(chunks,chunks_with_annotations)
        #print(no_class_chunks)
        tmp_row = sub_df.iloc[0]
        for chunk_off in no_class_chunks:
                tmp_row["OFFSET"] = chunk_off
                tmp_row["START_TIME"] = chunk_off
                tmp_row["END_TIME"] = chunk_off + chunk
                tmp_row["DURATION"] = chunk
                tmp_row["MANUAL ID"] = "no class"
                tmp_row["CHUNK_ID"] = chunk_off
                tmp_row["SPLIT"] = []
                
                new_df = new_df.append(tmp_row)
    df = new_df.sort_values(["IN FILE", "OFFSET"])
    return df
    
    
    