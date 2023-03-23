import pandas as pd
import os
import numpy as np
from annotation_post_processing import *

embeddings_df = pd.read_csv("./input/umap_cosmos_embeddings.csv")

unique_species = embeddings_df["FILE SPECIES"].unique()
print("# unique species: " + str(len(unique_species)))

from hdbscan import HDBSCAN
import pickle

print("Done with data loading!")

def hdbscan_model(embeddings:pd.DataFrame, embeddingColumns:list):
    np.random.seed(42)
    model = HDBSCAN(min_cluster_size = 10, min_samples = 5, cluster_selection_epsilon = 0.5, cluster_selection_method = "leaf")
    model.fit(embeddings[embeddingColumns])
    embeddings["LABELS"] = model.labels_
    embeddings.to_csv(f"./ClusteringModels/umap_general.csv")

hdbscan_model(embeddings_df, ["UMAP_0", "UMAP_1"])
