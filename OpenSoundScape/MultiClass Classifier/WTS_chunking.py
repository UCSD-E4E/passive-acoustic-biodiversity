import pandas as pd
import numpy as np

def annotation_chunker_no_duplicates(kaleidoscope_df, chunk_length, include_no_bird=False, bird=None):
    """
    Function that converts a Kaleidoscope-formatted Dataframe containing 
    annotations to uniform chunks of chunk_length. If there
    are mutliple bird species in the same clip, this function creates chunks
    for the more confident bird species.

    Note: if all or part of an annotation covers the last < chunk_length
    seconds of a clip it will be ignored. If two annotations overlap in 
    the same 3 second chunk, both are represented in that chunk
    Args:
        kaleidoscope_df (Dataframe)
            - Dataframe of annotations in kaleidoscope format

        chunk_length (int)
            - duration to set all annotation chunks
    Returns:
        Dataframe of labels with chunk_length duration 
        (elements in "OFFSET" are divisible by chunk_length).
    """

    #Init list of clips to cycle through and output dataframe
    #kaleidoscope_df["FILEPATH"] =  kaleidoscope_df["FOLDER"] + kaleidoscope_df["IN FILE"] 
    kaleidoscope_df['FILEPATH'] = kaleidoscope_df.loc[:,['FOLDER','IN FILE']].sum(axis=1)
    clips = kaleidoscope_df["FILEPATH"].unique()
    df_columns = {'FOLDER': 'str', 'IN FILE' :'str', 'CLIP LENGTH' : 'float64', 'CHANNEL' : 'int64', 'OFFSET' : 'float64',
                'DURATION' : 'float64', 'SAMPLE RATE' : 'int64','MANUAL ID' : 'str'}
    output_df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in df_columns.items()})
    
    # going through each clip
    for clip in clips:
        clip_df = kaleidoscope_df[kaleidoscope_df["FILEPATH"] == clip]
        path = clip_df["FOLDER"].unique()[0]
        file = clip_df["IN FILE"].unique()[0]
        birds = clip_df["MANUAL ID"].unique()
        sr = clip_df["SAMPLE RATE"].unique()[0]
        clip_len = clip_df["CLIP LENGTH"].unique()[0]

        # quick data sanitization to remove very short clips
        # do not consider any chunk that is less than chunk_length
        if clip_len < chunk_length:
            continue
        potential_annotation_count = int(clip_len)//int(chunk_length)

        # going through each species that was ID'ed in the clip
        arr_len = int(clip_len*1000)
        species_df = clip_df#[clip_df["MANUAL ID"] == bird]
        human_arr = np.zeros((arr_len))
        # looping through each annotation
        #print("========================================")
        for annotation in species_df.index:
            #print(species_df["OFFSET"][annotation])
            minval = int(round(species_df["OFFSET"][annotation] * 1000, 0))
            # Determining the end of a human label
            maxval = int(
                round(
                    (species_df["OFFSET"][annotation] +
                        species_df["DURATION"][annotation]) *
                    1000,
                    0))
            # Placing the label relative to the clip
            human_arr[minval:maxval] = 1
        # performing the chunk isolation technique on the human array

        for index in range(potential_annotation_count):
            #print("=======================")
            #print("-----------------------------------------")
            #print(index)
            chunk_start = index * (chunk_length*1000)
            chunk_end = min((index+1)*chunk_length*1000,arr_len)
            chunk = human_arr[int(chunk_start):int(chunk_end)]
            if max(chunk) >= 0.5:
                #Get row data
                row = pd.DataFrame(index = [0])
                annotation_start = chunk_start / 1000

                #Handle birdnet output edge case
                #print("-------------------------------------------")
                #print(sum(clip_df["DURATION"] == 3))
                #print(sum(clip_df["DURATION"] == 3)/clip_df.shape[0])
                #print("-------------------------------------------")
                if(sum(clip_df["DURATION"] == 3)/clip_df.shape[0] == 1):
                    #print("Processing here duration")
                    overlap = (clip_df["OFFSET"]+0.5 >= (annotation_start)) & (clip_df["OFFSET"]-0.5 <= (annotation_start))
                    annotation_df = clip_df[overlap]
                    #print(annotation_start, np.array(clip_df["OFFSET"]), overlap)
                    #print(annotation_df)
                else:
                    #print("Processing here")
                    overlap = is_overlap(clip_df["OFFSET"], clip_df["OFFSET"] + clip_df["DURATION"], annotation_start, annotation_start + chunk_length)
                    #print(overlap)
                    annotation_df = clip_df[overlap]
                    #print(annotation_df)
                
                #updating the dictionary
                if ('CONFIDENCE' in clip_df.columns):
                    annotation_df = annotation_df.sort_values(by="CONFIDENCE", ascending=False)
                    row["CONFIDENCE"] = annotation_df.iloc[0]["CONFIDENCE"]
                else:
                    #The case of manual id, or there is an annotation with no known confidence
                    row["CONFIDENCE"] = 1
                row["FOLDER"] = path
                row["IN FILE"] = file
                row["CLIP LENGTH"] = clip_len
                row["OFFSET"] = annotation_start
                row["DURATION"] = chunk_length
                row["SAMPLE RATE"] = sr
                row["MANUAL ID"] = annotation_df.iloc[0]["MANUAL ID"] 
                row["CHANNEL"] = 0
                output_df = pd.concat([output_df,row], ignore_index=True)
            elif(include_no_bird):
                #print(max(chunk))
                #Get row data
                row = pd.DataFrame(index = [0])
                annotation_start = chunk_start / 1000

                #updating the dictionary
                row["CONFIDENCE"] = 0
                row["FOLDER"] = path
                row["IN FILE"] = file
                row["CLIP LENGTH"] = clip_len
                row["OFFSET"] = annotation_start
                row["DURATION"] = chunk_length
                row["SAMPLE RATE"] = sr
                row["MANUAL ID"] = "no bird"
                row["CHANNEL"] = 0
                output_df = pd.concat([output_df,row], ignore_index=True)
    
    return output_df


def is_overlap(offset_df, end_df, chunk_start, chunk_end):
    is_both_before = (chunk_end < offset_df) & (chunk_start < offset_df)
    is_both_after = (end_df < chunk_end) & (end_df < chunk_start)
    return (~is_both_before) & (~is_both_after)
    
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import shutil, os
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
    
    
    