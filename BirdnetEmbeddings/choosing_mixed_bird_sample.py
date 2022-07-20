import pandas as pd
import shutil
import os

labels = pd.read_csv("mixed_bird_manual.csv")
file_names = pd.DataFrame(labels["IN FILE"].unique())
file_names.columns = ["IN FILE"]
file_names = file_names.assign(SPECIES = file_names["IN FILE"].apply(lambda x : " ".join(x.split("-")[:2])))
print(file_names)
print(f"Length of file_names: {len(file_names)}")

birdnet_species = pd.read_csv("birdnet_species.csv")
birdnet_species.columns = ["SPECIES"]
birdnet_species = birdnet_species["SPECIES"].apply(lambda x: x.split("_")[0]).tolist()
print(f"Length of birdnet_species: {len(birdnet_species)}")

file_names = file_names[file_names["SPECIES"].isin(birdnet_species)]
print(f"New length of file_names: {len(file_names)}")

sample_files = file_names["IN FILE"].sample(100, random_state = 1).tolist()
print(f"Length of sample_files: {sample_files}")

for file in sample_files:
    shutil.copy("../../Mixed_Bird/" + file, "./data/" + file)

sample_labels = labels[labels["IN FILE"].isin(sample_files)]
sample_labels.to_csv("mixed_bird_sample.csv")