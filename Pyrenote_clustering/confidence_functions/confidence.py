from .statistics import *
from .clustering import *

def confidence_is_100(df, users):
    return 100

def get_pairwise_iou(df, users):
    iou_scores = np.array([])
    df = df[df["LAST MOD BY"].isin(users)]

    for user in users:
        user_df = df[df["LAST MOD BY"] == user]
        not_users_df = df[df["LAST MOD BY"] != user]
        for not_user in not_users_df["LAST MOD BY"].unique():
            iou = clip_statistics(user_df,not_users_df,stats_type = "general", threshold = 0.5)["Global IoU"][0]
            iou_scores = np.append(iou_scores, iou)
    iou_scores = iou_scores.mean()
    return iou_scores


def get_pairwise_iou(df, users):
    iou_scores = np.array([])
    df = df[df["LAST MOD BY"].isin(users)]

    for user in users:
        user_df = df[df["LAST MOD BY"] == user]
        not_users_df = df[df["LAST MOD BY"] != user]
        for not_user in not_users_df["LAST MOD BY"].unique():
            iou = clip_statistics(user_df,not_users_df,stats_type = "general", threshold = 0.5)["Global IoU"][0]
            iou_scores = np.append(iou_scores, iou)
    iou_scores = iou_scores.mean()
    return iou_scores

def get_silhoutte_confidence(df,users):
    df = df[df["LAST MOD BY"].isin(users)]
    for file in df["IN FILE"].unique():
        model, clusters,  data_processed, silhoutte = label_clusters(df, DBSCAN_auto_dis_builder_min_dis2, file, distance = 1/2, verbose=False)
        return silhoutte[0]

def get_silhoutte_users_confidence(df,users):
    df = df[df["LAST MOD BY"].isin(users)]
    for file in df["IN FILE"].unique():
        model, clusters,  data_processed, silhoutte = label_clusters(df, DBSCAN_auto_dis_builder_min_dis2, file, distance = 1/2, verbose=False)
        return silhoutte[1]
