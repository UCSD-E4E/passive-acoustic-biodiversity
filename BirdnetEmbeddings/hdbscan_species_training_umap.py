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
        spec_embeddings.to_csv(f"./ClusteringModels/umap_species_specific/{species}.csv")
        i += 1
        print(f"Done with {i} of {len(unique_species)}")

hdbscan_model(embeddings_df, ["UMAP_0", "UMAP_1"])
