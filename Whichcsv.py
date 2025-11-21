#Selecting Clusters
import argparse
import math
import pyLCIO
import csv
from pyLCIO import UTIL, EVENT
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import ast

X, Y, Z = 0, 1, 2
TRACK_COL = "SiTracks"
CLUSTER_COL = "PandoraClusters"
PFO_COL = "PandoraPFOs"
BFIELD = 5.0
FACTOR = 3e-4

CAL_COLLECTIONS = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi",
    "YokeBarrelCollection",
    "YokeEndcapCollection"
]


system2name = {
    20: "ecal barrel",
    29: "ecal endcap",
    10: "hcal barrel",
    11: "hcal barrel",
    13: "yoke barrel",
    14: "yoke endcap"
    #Problem some are 13 and 14
}


#energies=[150]
#energylength = len(energies)
#rows = []
#Make a list of np.arrays
#tmp = pd.read_csv(f"outputpion_reco50_clusters.csv")
#for cat, row in tmp.iterrows(): #going through each event
#    event = row["event"]    
#    cluster_num = row["cluster_num"]
#    energy = row["energy"]
#    theta = row["theta"]
#    phi = row["phi"]
#    position = row["position"]
#    num_hit = row["num_hits"]
#    pos_tuple = ast.literal_eval(row["position"])
#    pos_vec = np.array(pos_tuple, dtype="float")
#    rows.append({
#        "energy": 10,
#        "event": event,
#        "cluster": cluster_num,
#        "vector": pos_vec
#    })
#df = pd.DataFrame(rows)

#def angle_between(vector1, vector2):
#    dot_product = np.dot(vector1, vector2)
#    norm_vec1 = np.linalg.norm(vector1)
#    norm_vec2 = np.linalg.norm(vector2)
#    if norm_vec1 == 0 or norm_vec2 == 0:
#        return np.nan  # Or raise an error, depending on desired behavior
#    cosine_angle = dot_product / (norm_vec1 * norm_vec2)
#    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
#    angle_radians = np.arccos(cosine_angle)
#    return angle_radians

#angles = []
#for (E, evt), group in df.groupby(["energy", "event"]):
#    vectors = group["vector"].to_list()
#    n = len(vectors)
#    if n > 1:
#        for i in range(n):
#            for j in range(i+1, n):
#                angle = angle_between(vectors[i], vectors[j])
#                angles.append({
#                    "energy": E,
#                    "event": evt,
#                    "angle_rad": angle,
#                    "angle_deg": np.degrees(angle)
#                })
#events_list = []
#for angle in angles:
#    if 0.17 < angle["angle_rad"] < 0.23:
#        events_list.append(angle["event"])

#print(events_list)

#Now printing and recording the hits to a csv for event 10
df = pd.read_csv(f"outputpion_reco10_hits.csv")

CAL_COLLECTIONS = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi",
    "YokeBarrelCollection",
    "YokeEndcapCollection"
]
all_hits = []
Event10=[13, 78, 80, 138, 170, 244, 313, 364, 365, 367, 395, 410, 443, 446, 447, 500, 501, 560, 585, 592, 636, 649, 655, 677, 693, 698, 706, 741, 776, 903, 929, 931, 987, 992]
Event50=[14, 29, 34, 104, 131, 154, 168, 238, 251, 320, 367, 391, 455, 508, 552, 564, 603, 727, 730, 761, 794, 812, 892, 909, 948, 976, 990]
Event100=[20, 51, 85, 154, 216, 232, 242, 353, 496, 497, 688, 825, 846, 945, 982]
Event150=[68, 181, 217, 254, 281, 297, 370, 470, 497, 512, 513, 639, 695, 768, 798, 869, 969, 970, 995]
Event200=[75, 237, 328, 352, 375, 559, 586, 728, 748, 758, 786, 820, 821, 835, 880, 934]
for cat, row in df.iterrows(): #going through each event
    if row["event"] in Event10:
    #if row["event"] == 100:
        hit_info = {
        "cluster_num": row["cluster_num"],
        "event": row["event"],
        "hit_num": row["hit_num"],
        "x": row["x"],
        "y": row["y"],
        "z": row["z"],
        "energy": row["energy"],
        "time": row["time"],
        "cell_id": row["cell_id"],
        "system": row["system"].lower().strip().replace(",", ""),
        "layer": row["layer"]
        }
        all_hits.append(hit_info)




with open("outputpion_reco10_hitswide.csv", "w", newline="") as f_hits:
    hits_fieldnames = [
        "event", "cluster_num", "hit_num", "x", "y", "z", "energy", "time", "cell_id", "system", "layer"
    ]
    writer = csv.DictWriter(f_hits, fieldnames=hits_fieldnames)
    writer.writeheader()
    for hit in all_hits:
        writer.writerow(hit)
