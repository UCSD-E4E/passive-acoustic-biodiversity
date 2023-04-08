from PyHa.statistics import *
from PyHa.IsoAutio import *
from PyHa.visualizations import *
from PyHa.annotation_post_processing import *
import pandas as pd
from joblib import Parallel, delayed
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

isolation_parameters_list = [isolation_parameters_microfaune_base,isolation_parameters_tweetynet_base, isolation_parameters_microfaune_filtering_1,
                                        isolation_parameters_microfaune_filtering_2, isolation_parameters_microfaune_filtering_3,isolation_parameters_tweetynet_base, isolation_parameters_tweetynet_filtering_1,
                                        isolation_parameters_tweetynet_filtering_2, isolation_parameters_tweetynet_filtering_3]



print("Loading in the Ground Truth")
# Loading in the Screaming Piha Dataset Ground Truth Labels
ground_truth = pd.read_csv("ScreamingPiha_Manual_Labels.csv")
ground_truth_3s = annotation_chunker(ground_truth,3)

screaming_piha_dataset = "./TEST/"

print("Computing Automated Labels using Multiprocessing")
automated_dfs = Parallel(n_jobs=-2)(delayed(generate_automated_labels)(screaming_piha_dataset,params) for params in isolation_parameters_list)

automated_dfs = Parallel(n_jobs=-2)(delayed(annotation_chunker)(df,3) for df in automated_dfs)

print("Computing Clip-level Statistics using Multiprocessing")
stats_dfs = Parallel(n_jobs=-2)(delayed(automated_labeling_statistics)(df,ground_truth_3s,stats_type="general") for df in automated_dfs)

print("Computing Global Statistics using Multiprocessing")
global_stats_dfs = Parallel(n_jobs=-2)(delayed(global_statistics)(df) for df in stats_dfs)

screaming_piha_results = pd.concat(global_stats_dfs)
screaming_piha_results.to_csv("Screaming_piha_local_score_filtering_experiments_parallel.csv",index=False)
toc = time.perf_counter()

print("Running in Parallel took " + str(toc-tic) + " seconds!")