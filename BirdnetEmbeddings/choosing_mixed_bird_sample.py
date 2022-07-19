import pandas as pd
import shutil

labels = pd.read_csv("mixed_bird_manual.csv")
file_names = pd.Series(labels["IN FILE"].unique())

sample_files = file_names.sample(100, random_state = 1)
print(sample_files)

for file in sample_files:
    shutil.copy("../../Mixed_Bird/" + file, "./data/" + file)

sample_labels = labels[labels["IN FILE"].isin(sample_files)]
sample_labels.to_csv("mixed_bird_sample.csv")