import random
import pandas as pd
import random
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import DBSCAN
import math
import statistics
from sklearn import metrics
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.cluster import KMeans
import hdbscan

from numpy import sin, cos, pi, linspace
def distance_cal2(s1,e1,s2,e2):
  return math.sqrt((s2 - s1) * (s2 - s1) + (e2 - e1) * (e2 - e1) )

def distance_cal3(s1,e1,s2,e2, d1, d2):
  return math.sqrt((s2 - s1) * (s2 - s1) + (e2 - e1) * (e2 - e1) + (d2 - d1) * (d2 - d1)) 


def label_clusters(data, model_builder, file_, distance=1/2, agreement=1, duration=True,  iterate_users=False, n=1, verbose=False):
  record = []
  data_oi = data[data["IN FILE"] == file_]
  users = data_oi["LAST MOD BY"].unique()
  record = []
  record_users = []


  if (iterate_users) :

      users = list(data_oi["LAST MOD BY"].unique())
      user_test = []
      count = 0;
      for i in range(1, len(users)+1):
        silhoutte_avg = 0
        silhoutte_avg_users = 0
        for j in range(0,n):
          user_test = (random.sample(users, i)[0:i])
          data_test = data_oi[data_oi['LAST MOD BY'].isin(user_test)]
          m,b,c, silhoutte = run_clustering(model_builder, data_test, user_test, distance, agreement, duration,  figure = i, verbose=verbose)
          silhoutte_avg = silhoutte_avg + silhoutte[0]
          silhoutte_avg_users = silhoutte_avg_users + silhoutte[1]
        record.append(silhoutte_avg/n)
        record_users.append(silhoutte_avg_users/n)
        print("mean silhoutte score for each # of sampled users from 1 to all users over 100 samples:", record)
        print("mean silhoutte score + unique_users/all users over 100 random samples for each # users:", record_users )
  
       
  else:
    return run_clustering(model_builder, data_oi, users, distance, agreement, duration,  figure = 1, verbose=verbose)
  
def run_clustering(model_builder, data_oi, users, distance=1/2, agreement=1, duration=True,  figure=1, verbose=False):
 
  data_oi["END TIMES"] = data_oi["DURATION"].add(data_oi["OFFSET"], fill_value=0)
  #print(tabulate(data_oi, headers='keys', tablefmt='psql'))

  neighborhood_size, model = model_builder(data = data_oi, distance = distance, users = users, agreement = agreement)
  if verbose:
    print("neighborhood size: ", neighborhood_size)

  model = model.fit(data_oi[["OFFSET", "END TIMES"]])
  clusters = model.fit_predict(data_oi[["OFFSET", "END TIMES"]])
  if (duration): clusters = model.fit_predict(data_oi[["OFFSET", "END TIMES", "DURATION"]])
  data_oi["cluster"] = clusters
 

  adv_cluster_count = 0
  adv_num_unique_users = 0 
  for i in range(max(clusters)):
     temp = data_oi[data_oi["cluster"] == i]
     adv_cluster_count += len(temp)
     adv_num_unique_users += len(pd.unique(temp['LAST MOD BY']))
     #print(get_longest_distance(temp, "OFFSET", "END TIMES"))
  adv_cluster_count /= int(max(clusters) + 1)
  adv_num_unique_users /=  int(max(clusters) + 1) #TEMP FIX INVESTIAGE HERE


  if (verbose):       
    print(clusters)
    print("adverage cluster size: ", adv_cluster_count)
    print("adverage unqiue users per cluster size: ", adv_num_unique_users)
  
  silhoutte = 0
  silhoutte_users = 0
  try:
    vr = metrics.calinski_harabasz_score(data_oi[["OFFSET", "END TIMES", "DURATION"]], clusters)
    silhoutte = metrics.silhouette_score(data_oi[["OFFSET", "END TIMES", "DURATION"]], clusters)
    silhoutte = (silhoutte + 1 )/2
    silhoutte_users = (silhoutte + adv_num_unique_users/len(users))/2

    if (verbose):  
      print("Variance Ratio Criterion", vr) 
      print("Note that VRC is less for DBSCAN in general")
      print("========================================") 
      print("Silhoutte Score              : ",silhoutte )
      print("Silhoutte Score scaled 0 - 1 : ",(silhoutte + 1 )/2)
      print("scaled avg Silhoutte users   : ",((silhoutte + 1 )/2+adv_num_unique_users/len(users))/2)
      
  except:
    if (verbose):  
      print("ERROR: not enough clusters to create meterics")

  if (verbose):
    colors = (plt.cm.rainbow(np.linspace(0, 1, max(clusters)+2)))
    i = 0
    for cluster in clusters:
      user_annotations = data_oi[data_oi["cluster"] == cluster]
      i += 1
      x = user_annotations["OFFSET"]
      y = user_annotations["END TIMES"]
      

      ##f = plt.figure(1)
      ##plt.xlabel("Start Time")
      ##plt.ylabel("Duration")
      ##plt.plot(x, y, 'o', color=colors[cluster+1]);
      #f.show()
      
      g = plt.figure(figure)
      plt.xlabel("Start Time")
      plt.ylabel("End Time")
      draw_circle(user_annotations, color=colors[cluster+1], alpha=1/len(x)*0.1, radius=neighborhood_size)
      plt.plot(x, y, 'o', color=colors[cluster+1], alpha = 1);
      
      g.show()
  return model, clusters, data_oi, (silhoutte, silhoutte_users)

 


def DBSCAN_auto_dis_builder_min_dis2(data = None, distance = 1, users = None, agreement = 0.5, duration=False):
    NEIGHBORHOOD_SCALAR = distance

    n = 0
    adv_distance = []
    dists_raw = []
    for i in range(len(users)):
      user_labels = data[data['LAST MOD BY'] == users[i]]
      s1 = 0
      e1 = 0
      s2 = 0
      e2 = 0
      d1 = 0
      d2 = 0
      skip = True
      for index, row in user_labels.iterrows():
        #print(s1,e1,s2,e2)
        s2 = float(row["OFFSET"])
        e2 = float(row["END TIMES"])
        d2 = float(row["DURATION"])
        dist = distance_cal3(s1,e1,s2,e2, d1, d2)
        if (not skip):
          dists_raw.append(dist)
        
        skip = False
        s1 = s2
        e1 = e2
        d1 = d2

    if len(dists_raw) == 0:
      dists_raw.append(1) #TODO: Investigate edge case
    adv_distance = min(dists_raw) #* NEIGHBORHOOD_SCALAR #
    return adv_distance, DBSCAN(
                    eps=adv_distance*0.9, 
                    min_samples=2,
                  
                  )

def HDBSCAN_builder(data = None, distance = 1, users = None, agreement = 0.5, duration=False):
    return 0, hdbscan.HDBSCAN(min_cluster_size=1)