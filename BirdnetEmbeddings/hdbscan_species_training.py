import pandas as pd
import os
import numpy as np
from annotation_post_processing import *

embeddingColumns = [str(i) for i in range(420)]
columnNames = ["START", "END"] + embeddingColumns
path = './input/cosmos_embeddings/'

def generate_embeddings_from_file(path, filename:str):
    with open(path + filename, 'r') as f:
        data = f.read()
    with open(path + filename, 'w') as f:
        f.write(",".join(data.split("\t")))
    file_df = pd.read_csv(path + filename, names = columnNames)
    file_df["IN FILE"] = filename[:filename.index(".birdnet")] + ".wav"
    file_df["FILE SPECIES"] = filename.split(" - ")[-1].split(".")[0]
    return file_df

def generate_embeddings(path):
    df = pd.DataFrame()
    for filename in os.listdir(path):
        try:
            df = pd.concat([df, generate_embeddings_from_file(path, filename)], ignore_index = True)
            print("Done with " + filename)
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

unique_species = embeddings_df["FILE SPECIES"].unique()

from hdbscan import HDBSCAN
import pickle

print("Done with data loading!")

def hdbscan_model(embeddings:pd.DataFrame, embeddingColumns:list):
    np.random.seed(42)
    i = 0
    for species in unique_species:
        if (min(5, len(embeddings[embeddings["FILE SPECIES"] == species].index))) <= 1:
            continue
        model = HDBSCAN(min_cluster_size = min(5, len(embeddings[embeddings["FILE SPECIES"] == species].index)),
            min_samples = min(5, len(embeddings[embeddings["FILE SPECIES"] == species].index)),
            cluster_selection_epsilon = 0.5,
            cluster_selection_method = "leaf",
        )
        spec_embeddings:pd.DataFrame = embeddings[embeddings["FILE SPECIES"] == species].copy()
        model.fit(spec_embeddings[embeddingColumns])
        spec_embeddings["LABELS"] = model.labels_
        spec_embeddings.to_csv(f"./ClusteringModels/species_specific/{species}.csv")
        i += 1
        print(f"Done with {i} of {len(unique_species)}")

hdbscan_model(embeddings_df, embeddingColumns)
