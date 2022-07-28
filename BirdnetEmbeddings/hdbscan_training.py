import pandas as pd
import os
import numpy as np
from annotation_post_processing import *
import hdbscan
from hdbscan import HDBSCAN
import pickle

embeddingColumns = [str(i) for i in range(420)]
columnNames = ["START", "END"] + embeddingColumns
path = './input/xc_embeddings/'

def generate_embeddings_from_file(path, filename):
    with open(path + filename, 'r') as f:
        data = f.read()
    with open(path + filename, 'w') as f:
        f.write(",".join(data.split("\t")))
    file_df = pd.read_csv(path + filename, names = columnNames)
    file_df["IN FILE"] = filename[:filename.index(".birdnet")] + ".wav"
    return file_df

def generate_embeddings(path):
    df = pd.DataFrame()
    for filename in os.listdir(path):
        try:
            df = pd.concat([df, generate_embeddings_from_file(path, filename)], ignore_index = True)
        except Exception as e:
            print("Something went wrong with: " + filename)
    df["PATH"] = path
    columns = df.columns.tolist()
    columns = columns[-2:] + columns[:-2]
    df = df[columns]
    df = df.sort_values(["IN FILE", "START"], ascending = True)
    df = df.reset_index(drop = True)
    return df

embeddings_df = generate_embeddings(path)
annotations_df = pd.read_csv("xc_annotations.csv")

# removing duplicate annotations
grouped_annotations = annotations_df.groupby(["IN FILE", "OFFSET"])["CONFIDENCE"].max()
annotations_df["MANUAL ID"] = annotations_df.apply(lambda x: x["MANUAL ID"] \
    if grouped_annotations.loc[x["IN FILE"], x["OFFSET"]] == x["CONFIDENCE"] else pd.NA, axis = 1)
annotations_df = annotations_df.dropna(subset = ["MANUAL ID"]).reset_index(drop = True)

manual_df = pd.read_csv("mixed_bird_sample.csv")
manual_df = pd.DataFrame(annotation_chunker(manual_df, 3))
manual_df["MANUAL ID"] = manual_df["MANUAL ID"].apply(lambda x: " ".join(x.split(" ")[:2]))

# Data cleaning to avoid file does not exist
intersection_files = list(set(embeddings_df["IN FILE"].unique()).intersection(set(annotations_df["IN FILE"])))
embeddings_df = embeddings_df[embeddings_df["IN FILE"].isin(intersection_files)]
annotations_df = annotations_df[annotations_df["IN FILE"].isin(intersection_files)]
manual_df = manual_df[manual_df["IN FILE"].isin(intersection_files)]

# Adding manual ids to embeddings
k = 0
def embed_id(x):
    filenames = manual_df["IN FILE"] == x["IN FILE"]
    offsets = np.isclose(manual_df["OFFSET"], x["START"])
    both = filenames & offsets
    if not np.any(both):
        return "No bird"
    return manual_df[both]["MANUAL ID"].iloc[0]

embeddings_df["MANUAL ID"] = embeddings_df.apply(embed_id, axis = 1)
embeddings_df["FILE SPECIES"] = embeddings_df["IN FILE"].apply(lambda x: " ".join(x.split("-")[:2]))

# To change all of the birdnet annotations manual ids to species names
birdnet_species = pd.read_csv("birdnet_species.csv")
birdnet_species.columns = ["SPECIES"]
birdnet_species = birdnet_species.assign(
    SPECIES = birdnet_species["SPECIES"].apply(lambda x: x.split("_")[0]), 
    COMMON = birdnet_species["SPECIES"].apply(lambda x: x.split("_")[1])
)
birdnet_species = birdnet_species.set_index("COMMON").to_dict()["SPECIES"]

annotations_df["MANUAL ID"] = annotations_df["MANUAL ID"].apply(lambda x: birdnet_species[x])

def hdbscan_model(embeddings:pd.DataFrame, embeddingColumns:list):
    np.random.seed(42)
    model = HDBSCAN(min_cluster_size = 10, min_samples = 1)
    model.fit(embeddings[embeddingColumns])
    pickle.dump(model, open(f"./ClusteringModels/hdbscan_model.pkl", "wb"))

hdbscan_model(embeddings_df, embeddingColumns)
