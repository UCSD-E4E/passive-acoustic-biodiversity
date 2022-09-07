import pandas as pd
import os
import numpy as np
from annotation_post_processing import *

embeddingColumns = [str(i) for i in range(420)]
columnNames = ["START", "END"] + embeddingColumns
path = './input/cosmos_embeddings/'

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

from hdbscan import HDBSCAN
import pickle

def hdbscan_model(embeddings:pd.DataFrame, embeddingColumns:list):
    np.random.seed(42)
    model = HDBSCAN(min_cluster_size = 5, min_samples = 1, cluster_selection_epsilon = 0.5, cluster_selection_method = "leaf")
    model.fit(embeddings[embeddingColumns])
    pickle.dump(model, open(f"./ClusteringModels/hdbscan_cosmos_model.pkl", "wb"))

hdbscan_model(embeddings_df, embeddingColumns)
