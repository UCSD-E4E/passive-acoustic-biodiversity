from PyHa.statistics import *
from PyHa.IsoAutio import *
from PyHa.visualizations import *
from PyHa.annotation_post_processing import *
import pandas as pd
import time

tic = time.perf_counter()
print("Establishing Isolation Parameters")

isolation_parameters_microfaune_base = {
    "model" : "microfaune",
    "technique" : "chunk",
    "threshold_const" : 2.0,
    "threshold_min" : 0.1,
    'threshold_type' : "median",
    "chunk_size" : 3.0,
    "verbose" : True
}

isolation_parameters_tweetynet_base = {
    "model" : "tweetynet",
    "tweety_output": True,
    "verbose" : True
}

isolation_parameters_microfaune_filtering_1 = {
    "model" : "microfaune",
    "technique" : "chunk",
    "threshold_const" : 2.0,
    "threshold_min" : 0.1,
    'threshold_type' : "median",
    "chunk_size" : 3.0,
    "filter_local_scores" : (0.08,15),
    "verbose" : True
}

isolation_parameters_tweetynet_filtering_1 = {
    "model" : "tweetynet",
    "tweety_output": True,
    "filter_local_scores" : (0.08,15),
    "verbose" : True
}

isolation_parameters_microfaune_filtering_2 = {
    "model" : "microfaune",
    "technique" : "chunk",
    "threshold_const" : 2.0,
    "threshold_min" : 0.1,
    'threshold_type' : "median",
    "chunk_size" : 3.0,
    "filter_local_scores" : (0.15,15),
    "verbose" : True
}

isolation_parameters_tweetynet_filtering_2 = {
    "model" : "tweetynet",
    "tweety_output": True,
    "filter_local_scores" : (0.15,15),
    "verbose" : True
}

isolation_parameters_microfaune_filtering_3 = {
    "model" : "microfaune",
    "technique" : "chunk",
    "threshold_const" : 2.0,
    "threshold_min" : 0.1,
    'threshold_type' : "median",
    "chunk_size" : 3.0,
    "filter_local_scores" : (0.08,25),
    "verbose" : True
}

isolation_parameters_tweetynet_filtering_3 = {
    "model" : "tweetynet",
    "tweety_output": True,
    "filter_local_scores" : (0.08,25),
    "verbose" : True
}

print("Loading in the Ground Truth")
# Loading in the Screaming Piha Dataset Ground Truth Labels
ground_truth = pd.read_csv("ScreamingPiha_Manual_Labels.csv")
ground_truth_3s = annotation_chunker(ground_truth,3)

screaming_piha_dataset = "./TEST/"

print("Computing Automated Labels")
# Collecting automated labels
automated_df_microfaune_base = generate_automated_labels(screaming_piha_dataset,isolation_parameters_microfaune_base)
automated_df_tweetynet_base = generate_automated_labels(screaming_piha_dataset,isolation_parameters_tweetynet_base)
automated_df_tweetynet_base = annotation_chunker(automated_df_tweetynet_base,3)
automated_df_microfaune_filtering_1 = generate_automated_labels(screaming_piha_dataset,isolation_parameters_microfaune_filtering_1)
automated_df_tweetynet_filtering_1 = generate_automated_labels(screaming_piha_dataset,isolation_parameters_tweetynet_filtering_1)
automated_df_tweetynet_filtering_1 = annotation_chunker(automated_df_tweetynet_filtering_1,3)
automated_df_microfaune_filtering_2 = generate_automated_labels(screaming_piha_dataset,isolation_parameters_microfaune_filtering_2)
automated_df_tweetynet_filtering_2 = generate_automated_labels(screaming_piha_dataset,isolation_parameters_tweetynet_filtering_2)
automated_df_tweetynet_filtering_2 = annotation_chunker(automated_df_tweetynet_filtering_2,3)
automated_df_microfaune_filtering_3 = generate_automated_labels(screaming_piha_dataset,isolation_parameters_microfaune_filtering_3)
automated_df_tweetynet_filtering_3 = generate_automated_labels(screaming_piha_dataset,isolation_parameters_tweetynet_filtering_3)
automated_df_tweetynet_filtering_3 = annotation_chunker(automated_df_tweetynet_filtering_3,3)

print("Computing Statistics Comparing Automated Labels to Ground Truth")
# Comparing automated labels to human labels
stats_df_microfaune_base = automated_labeling_statistics(automated_df_microfaune_base,ground_truth_3s,stats_type = "general")
stats_df_tweetynet_base = automated_labeling_statistics(automated_df_tweetynet_base,ground_truth_3s,stats_type = "general")
stats_df_microfaune_filtering_1 = automated_labeling_statistics(automated_df_microfaune_filtering_1,ground_truth_3s,stats_type = "general")
stats_df_tweetynet_filtering_1 = automated_labeling_statistics(automated_df_tweetynet_filtering_1,ground_truth_3s,stats_type = "general")
stats_df_microfaune_filtering_2 = automated_labeling_statistics(automated_df_microfaune_filtering_2,ground_truth_3s,stats_type = "general")
stats_df_tweetynet_filtering_2 = automated_labeling_statistics(automated_df_tweetynet_filtering_2,ground_truth_3s,stats_type = "general")
stats_df_microfaune_filtering_3 = automated_labeling_statistics(automated_df_microfaune_filtering_3,ground_truth_3s,stats_type = "general")
stats_df_tweetynet_filtering_3 = automated_labeling_statistics(automated_df_tweetynet_filtering_3,ground_truth_3s,stats_type = "general")

# Microfaune Stats
global_stats_df_microfaune_base = global_statistics(stats_df_microfaune_base)
global_stats_df_microfaune_base["Experiment"] = "microfaune_base"
global_stats_df_microfaune_filtering_1 = global_statistics(stats_df_microfaune_filtering_1)
global_stats_df_microfaune_filtering_1["Experiment"] = "microfaune_filtering_1"
global_stats_df_microfaune_filtering_2 = global_statistics(stats_df_microfaune_filtering_2)
global_stats_df_microfaune_filtering_2["Experiment"] = "microfaune_base_filtering_2"
global_stats_df_microfaune_filtering_3 = global_statistics(stats_df_microfaune_filtering_3)
global_stats_df_microfaune_filtering_3["Experiment"] = "microfaune_base_filtering_3"

# Tweetynet Stats
global_stats_df_tweetynet_base = global_statistics(stats_df_tweetynet_base)
global_stats_df_tweetynet_base["Experiment"] = "tweetynet_base"
global_stats_df_tweetynet_filtering_1 = global_statistics(stats_df_tweetynet_filtering_1)
global_stats_df_tweetynet_filtering_1["Experiment"] = "tweetynet_base"
global_stats_df_tweetynet_filtering_2 = global_statistics(stats_df_tweetynet_filtering_2)
global_stats_df_tweetynet_filtering_2["Experiment"] = "tweetynet_filtering_1"
global_stats_df_tweetynet_filtering_3 = global_statistics(stats_df_tweetynet_filtering_3)
global_stats_df_tweetynet_filtering_3["Experiment"] = "tweetynet_base_filtering_1"

combine = [global_stats_df_microfaune_base,global_stats_df_microfaune_filtering_1,global_stats_df_microfaune_filtering_2,global_stats_df_microfaune_filtering_3,
           global_stats_df_tweetynet_base,global_stats_df_tweetynet_filtering_1,global_stats_df_tweetynet_filtering_2,global_stats_df_tweetynet_filtering_3]

screaming_piha_results = pd.concat(combine)
screaming_piha_results.to_csv("Screaming_piha_local_score_filtering_experiments.csv",index=False)

toc = time.perf_counter()

print("Baseline took " + str(toc-tic) + " seconds!")