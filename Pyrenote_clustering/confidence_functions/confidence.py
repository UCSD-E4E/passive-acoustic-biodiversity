from .statistics import *
from .clustering import *
from .chunking import annotation_chunker, fast_chunker

def confidence_is_100(df, users):
    return 100

#NOTE TO SELF, CREATE BETTER IOU MERTIC SYSTEM

def get_pairwise_iou(df, users):
    iou_scores = np.array([])
    df = df[df["LAST MOD BY"].isin(users)]

    for user in users:
        user_df = df[df["LAST MOD BY"] == user]
        not_users_df = df[df["LAST MOD BY"] != user]
        species_list = np.unique(user_df["MANUAL ID"].append(not_users_df["MANUAL ID"]))
        for species in species_list:
            tmp_user_df = user_df[user_df["MANUAL ID"] == species]
            tmp_not_users_df = not_users_df[not_users_df["MANUAL ID"] == species]
            try:
                iou = clip_statistics(tmp_user_df,tmp_not_users_df,stats_type = "general", threshold = 0.5)["Global IoU"][0]
                #print("species", species, "user", user, "iou", iou)
                iou_scores = np.append(iou_scores, iou)
            except:
                #print("species", species, "user", user, "iou", 0)
                iou_scores = np.append(iou_scores, 0)
    iou_scores = iou_scores.mean()
    return iou_scores


def get_silhoutte_confidence(df,users):
    df = df[df["LAST MOD BY"].isin(users)]
    scores = np.array([])
    for file in df["IN FILE"].unique():
        for manual_id in df["MANUAL ID"].unique():
            
            tmp_df = df[df["MANUAL ID"] == manual_id]    
            model, clusters,  data_processed, silhoutte = label_clusters(tmp_df, DBSCAN_auto_dis_builder_min_dis2, file, distance = 1/2, verbose=False)
            #print("species", manual_id, "file", file, "score", silhoutte[0])
            scores = np.append(scores,  silhoutte[0])
    return scores.mean()

def get_silhoutte_users_confidence(df,users):
    df = df[df["LAST MOD BY"].isin(users)]
    scores = np.array([])
    for file in df["IN FILE"].unique():
        for manual_id in df["MANUAL ID"].unique():
            
            tmp_df = df[df["MANUAL ID"] == manual_id]    
            model, clusters,  data_processed, silhoutte = label_clusters(tmp_df, DBSCAN_auto_dis_builder_min_dis2, file, distance = 1/2, verbose=False)
            #print("species", manual_id, "file", file, "score", silhoutte[1])
            scores = np.append(scores,  silhoutte[1])
    return scores.mean()

#def majority_vote(df, users, chunk_length=1):
#    df = df[df["LAST MOD BY"].isin(users)]
#    df = annotation_chunker(df, chunk_length)
#    df = df.groupby(by=["LAST MOD BY", "OFFSET"]).max().reset_index()
#    #TODO ADD SPECIES STUFF HERE
#    df = df.groupby(by=["OFFSET"]).count().reset_index()
#    counts = (df["FOLDER"]/len(users)).mean() #Does mean make sense here?
#    return counts


def fast_majority_vote(df, users, chunk_length=3):
    df = df[df["LAST MOD BY"].isin(users)]
    df = fast_chunker(df, chunk_length)
    #print(df)
    df["LAST MOD BY"] = df["LAST MOD BY"].apply(lambda x: len(x.split(",")))
    counts = (df["LAST MOD BY"]/len(users)).mean()
    return  counts