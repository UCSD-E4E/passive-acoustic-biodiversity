import pandas as pd
import os
import numpy as np
from annotation_post_processing import *

filtered_embeddings = dict()

for a in [5, 10, 20, 50, 100, 200, 500]:
    for b in [5, 10, 20, 50, 100, 200, 500]:
        filtered_embeddings[a,b] = pd.read_csv(f"./input/filtered_embeddings_umap_hyper/{a}_{b}_filtered")

print("Created filter!")

count2 = 0
def create_annotation_filter(x: pd.Series, filter: pd.DataFrame) -> pd.DataFrame:
    filter_x = filter[filter["IN FILE"].str.startswith(x["IN FILE"].split(".mp3")[0])]
    starts = filter_x["START"].to_numpy()
    ends = filter_x["END"].to_numpy()
    close_starts = np.isclose(starts, x["OFFSET"]).sum()
    close_ends = np.isclose(ends, x["OFFSET"] + x["DURATION"]).sum()
    middle1 = starts < x["OFFSET"]
    middle2 = ends > x["OFFSET"] + x["DURATION"]
    middle = (middle1*middle2).sum()
    if (close_starts + close_ends + middle) > 0:
        x["FILTERED"] = True
    else:
        x["FILTERED"] = False
    global count2
    count2 += 1
    print(f"Completed {count2} annotations")
    return x

automated_dfs_split = []

for i in range(5):
    automated_dfs_split.append(pd.read_csv(f"./cosmos_annotations/split/{i}_automated_split"))

automated_filtered_all = dict()
for a in [200, 500]:
    for b in [200, 500]:
        automated_dfs_filtered = [df.apply(lambda x: create_annotation_filter(x, filtered_embeddings[a, b]), axis = 1) for df in automated_dfs_split]
        automated_dfs_filtered = [df[~df["FILTERED"]] for df in automated_dfs_filtered]
        automated_filtered_all[a,b] = automated_dfs_filtered
        count2 = 0

for a in [200, 500]:
    for b in [200, 500]:
        for i in range(5):
            automated_filtered_all[a,b][i].to_csv(f"./input/automated_filtered_embeddings_umap/{i}_{a}_{b}")

