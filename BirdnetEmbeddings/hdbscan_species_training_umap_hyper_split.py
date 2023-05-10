import pandas as pd
import os
import numpy as np
from annotation_post_processing import *

embeddingColumns = [str(i) for i in range(420)] + ["UMAP_0", "UMAP_1"]
columnNames = ["START", "END"] + embeddingColumns
path = './input/cosmos_embeddings/'

automated_dfs:list[pd.DataFrame] = []
automated_dfs.append(pd.read_csv("./cosmos_annotations/automated_cosmos_tweety_to_file.csv"))
automated_dfs.append(pd.read_csv("./cosmos_annotations/COSMOS_BirdNET-Lite_Labels_05Conf.csv"))
automated_dfs.append(pd.read_csv("./cosmos_annotations/COSMOS_BirdNET-Lite_Labels_100.csv"))
automated_dfs.append(pd.read_csv("./cosmos_annotations/COSMOS_BirdNET-Lite-Filename_Labels_05Conf.csv"))
automated_dfs.append(pd.read_csv("./cosmos_annotations/COSMOS_Microfaune-Filename_Labels_100.csv"))
print(automated_dfs)

automated_dfs_split:list[pd.DataFrame] = []
for i in range(5):
    automated_dfs_split.append(pd.read_csv(f"./cosmos_annotations/split/{i}_automated_split"))
print(automated_dfs_split)

# Results for general embedding clustering
# hdbscan_results = pd.read_csv("./ClusteringModels/umap_general.csv")

# Results for species-specific clustering
hdbscan_all = dict()
i = 0
for a in [5, 10, 20, 50, 100, 200, 500]:
    for b in [5, 10, 20, 50, 100, 200, 500]:
        hdbscan_all[a,b] = pd.read_csv(f"./input/filtered_embeddings_umap_hyper/{a}_{b}_filtered")

filtered_embeddings = hdbscan_all
print("Created filter")

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
    return x

automated_filtered_all = dict()
for a in [5, 10, 20, 50, 100, 200, 500]:
    for b in [5, 10, 20, 50, 100, 200, 500]:
        automated_dfs_filtered = [df.apply(lambda x: create_annotation_filter(x, filtered_embeddings[a, b]), axis = 1) for df in automated_dfs_split]
        automated_dfs_filtered = [df[~df["FILTERED"]] for df in automated_dfs_filtered]
        automated_filtered_all[a,b] = automated_dfs_filtered
        for i in range(5):
            automated_filtered_all[a,b][i].to_csv(f"./cosmos_annotations/filtered/species_{i}/{a}_{b}_filtered.csv")
        print(f"Done with filtering for hyperparameters {a} and {b}")
print(automated_filtered_all[5,5])

print("Done with filtering!")

print([df.shape[0] for df in automated_dfs])
print([df.shape[0] for df in automated_dfs_split])
print([df.shape[0] for df in automated_filtered_all[5,5]])

from statistics_1 import *

manual_df = pd.read_csv("cosmos_annotations/cosmos_labeled_data_files_added.csv")
manual_df["IN FILE"] = manual_df["IN FILE"].apply(lambda x: " ".join(x.split("_")))
manual_df["FOLDER"] = "./cosmos_annotations/"

import warnings
warnings.filterwarnings("ignore")
clip_stats_original = [clip_statistics(df, manual_df, "general") for df in automated_dfs]
clip_stats_filtered_all = dict()
for a in [5, 10, 20, 50, 100, 200, 500]:
    for b in [5, 10, 20, 50, 100, 200, 500]:
        clip_stats_filtered_all[a,b] = [clip_statistics(df, manual_df, "general") for df in automated_filtered_all[a,b]]

class_stats_original = [class_statistics(stats) for stats in clip_stats_original]
class_stats_filtered_all = dict()
for a in [5, 10, 20, 50, 100, 200, 500]:
    for b in [5, 10, 20, 50, 100, 200, 500]:
        class_stats_filtered_all[a,b] = [class_statistics(stats) for stats in clip_stats_filtered_all[a,b]]

for i in range(5):
    class_stats_original[i].to_csv(f"./cosmos_annotations/stats/{i}_original.csv")
    for a in [5, 10, 20, 50, 100, 200, 500]:
        for b in [5, 10, 20, 50, 100, 200, 500]:
            class_stats_filtered_all[a,b][i].to_csv(f"./cosmos_annotations/stats/{i}_{a}_{b}_filtered.csv")
            
print("Uploaded all stats!")