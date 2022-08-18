import pandas as pd
import numpy as np

# For backup
count1 = 0
def split_annotations(df: pd.DataFrame):
    all_split_ann = pd.DataFrame(columns=df.columns)
    for i in range(df.shape[0]):
        split_ann = pd.DataFrame(df.iloc[i]).T
        while True:
            ind = split_ann.shape[0] - 1
            row = split_ann.index[ind]
            if (int(split_ann.at[row,"OFFSET"] / 3) == int((split_ann.at[row,"OFFSET"] + split_ann.at[row,"DURATION"])/3)):
                break
            new_x = pd.DataFrame(split_ann.iloc[ind].copy()).T
            new_x.index = [row + 1]
            new_x["OFFSET"] = 3.0 * (int(split_ann.at[row,"OFFSET"] / 3) + 1)
            split_ann.at[row,"DURATION"] = 3.0 * (int(split_ann.at[row,"OFFSET"] / 3) + 1) - split_ann.at[row,"OFFSET"]
            split_ann = pd.concat([split_ann, new_x])
        all_split_ann = pd.concat([all_split_ann, split_ann])
        global count1
        count1 += 1
        print(f"Completed {count1} annotations")
    return all_split_ann

count2 = 0
def create_annotation_filter(x: pd.Series, filter: pd.DataFrame) -> pd.DataFrame:
    filter_x = filter[filter["IN FILE"].str.startswith(x["IN FILE"].split(".mp3")[0])][["START", "END"]]
    for i in range(len(filter_x["START"])):
        start, end = filter_x.iloc[i, 0], filter_x.iloc[i, 1]
        offset = x["OFFSET"]
        duration = x["DURATION"]
        if (start <= offset <= end):
            x["OFFSET"] = end
        if (start <= offset + duration <= end):
            x["DURATION"] = start - x["OFFSET"]
        if x["DURATION"] < 0:
            x["FILTERED"] = True
            break
    if (np.isclose(x["DURATION"], 0) or x["DURATION"] < 0):
        x["FILTERED"] = True
    else:
        x["FILTERED"] = False
    global count2
    count2 += 1
    print(f"Completed {count2} annotations")
    return x