import pandas as pd
import os
import numpy as np
from annotation_post_processing import *

embeddingColumns = [str(i) for i in range(420)] + ["UMAP_0", "UMAP_1"]
columnNames = ["START", "END"] + embeddingColumns

embeddings_df = pd.read_csv("./input/umap_cosmos_embeddings.csv")

unique_species = embeddings_df["FILE SPECIES"].unique()
print("# unique species: " + str(len(unique_species)))

automated_dfs:list[pd.DataFrame] = []
automated_dfs.append(pd.read_csv("./cosmos_annotations/automated_cosmos_tweety_to_file.csv"))
automated_dfs.append(pd.read_csv("./cosmos_annotations/COSMOS_BirdNET-Lite_Labels_05Conf.csv"))
automated_dfs.append(pd.read_csv("./cosmos_annotations/COSMOS_BirdNET-Lite_Labels_100.csv"))
automated_dfs.append(pd.read_csv("./cosmos_annotations/COSMOS_BirdNET-Lite-Filename_Labels_05Conf.csv"))
automated_dfs.append(pd.read_csv("./cosmos_annotations/COSMOS_Microfaune-Filename_Labels_100.csv"))
print(automated_dfs)

from hdbscan import HDBSCAN

print("Done with data loading!")

def hdbscan_model(embeddings:pd.DataFrame, embeddingColumns:list):
    np.random.seed(42)
    i = 0
    for species in unique_species:
        if (min(5, len(embeddings[embeddings["FILE SPECIES"] == species].index))) <= 1:
            continue
        j = 0
        for a in [5, 10, 20, 50, 100, 200, 500]:
            for b in [5, 10, 20, 50, 100, 200, 500]:
                model = HDBSCAN(min_cluster_size = min(a, len(embeddings[embeddings["FILE SPECIES"] == species].index)),
                    min_samples = min(b, len(embeddings[embeddings["FILE SPECIES"] == species].index)),
                    cluster_selection_epsilon = 0.5,
                    cluster_selection_method = "leaf",
                )
                spec_embeddings:pd.DataFrame = embeddings[embeddings["FILE SPECIES"] == species].copy()
                model.fit(spec_embeddings[embeddingColumns])
                spec_embeddings["LABELS"] = model.labels_
                spec_embeddings.to_csv(f"./ClusteringModels/umap_species_specific_hyper/{a}_{b}_{species}.csv")
                j += 1
                print(f"Done with {j} iterations of hyperparameters for {species}")
        i += 1
        print(f"Done with {i} of {len(unique_species)}")

hdbscan_model(embeddings_df, ["UMAP_0", "UMAP_1"])

# Results for general embedding clustering
# hdbscan_results = pd.read_csv("./ClusteringModels/umap_general.csv")

# Results for species-specific clustering
hdbscan_all = dict()
unique_species = embeddings_df["FILE SPECIES"].unique()
i = 0
for a in [5, 10, 20, 50, 100, 200, 500]:
    for b in [5, 10, 20, 50, 100, 200, 500]:
        hdbscan_results = pd.DataFrame(columns=["FILE SPECIES", "PATH"] + columnNames + ["IN FILE", "LABELS"])
        j = 0
        for species in unique_species:
            species_result = pd.read_csv(f"./ClusteringModels/umap_species_specific_hyper/{a}_{b}_{species}.csv").drop(["Unnamed: 0"], axis=1)
            # Method 1: Simply filters out what was labeled as noise in recording
            filter_1 = species_result[species_result["LABELS"] == -1]
            
            # Method 2: Filters out noise and creates the filter by checking the n most frequent values of embedding labels (essentially to see most frequent bird labels, should be dominant)
            n = 2
            species_result = species_result[species_result["LABELS"] != -1]
            max_nums = species_result["LABELS"].value_counts()[:n].index.tolist() # picking n most frequent values
            filter_2 = species_result[~species_result["LABELS"].isin(max_nums)]
            
            # filter = filter_1
            filter = pd.concat([filter_1, filter_2], axis=0)
            
            hdbscan_results = pd.concat([hdbscan_results, filter], axis=0)
            j += 1
            print(f"Done with {j} of {len(unique_species)}")
        hdbscan_all[a,b] = hdbscan_results.reset_index(drop = True)
        i += 1
        print(f"Done with {i} iterations of hyperparameters")
        
filtered_embeddings = hdbscan_all

for a in [5, 10, 20, 50, 100, 200, 500]:
    for b in [5, 10, 20, 50, 100, 200, 500]:
        filtered_embeddings[a,b].to_csv(f"./input/filtered_embeddings_umap_hyper/{a}_{b}_filtered")

print("Created filter!")

count1 = 0
def split_annotations(df: pd.DataFrame):
    all_split_ann = pd.DataFrame(columns = df.columns)
    for i in range(df.shape[0]):
        x = df.iloc[i]
        startsends = np.linspace(3.0 * (int(x["OFFSET"] / 3)), 3.0 * (int((x["OFFSET"] + x["DURATION"])/ 3) + 1), int((x["OFFSET"] + x["DURATION"])/ 3) - int(x["OFFSET"] / 3) + 2)
        starts = startsends[:-1]
        starts[0] = x["OFFSET"]
        ends = startsends[1:]
        ends[-1] = x["OFFSET"] + x["DURATION"]
        split_ann = pd.DataFrame(columns = x.index)
        for i in range(len(starts)):
            new_x = pd.DataFrame(x.copy()).T
            new_x["OFFSET"] = starts[i]
            new_x["DURATION"] = ends[i] - starts[i]
            if np.isclose(new_x["DURATION"], 0):
                continue
            split_ann = pd.concat([split_ann, new_x])
        all_split_ann = pd.concat([all_split_ann, split_ann])
        global count1
        count1 += 1
        print(f"Completed {count1} annotations")
    return all_split_ann.reset_index(drop = True)

automated_dfs_split = [split_annotations(df) for df in automated_dfs]

for i in range(5):
    automated_dfs_split[i].to_csv(f"./cosmos_annotations/split/{i}_automated_split")
