#Getting the angle between clusters

import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import ast
from collections import defaultdict

CAL_COLLECTIONS = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi",
    "YokeBarrelCollection",
    "YokeEndcapCollection"
]

energies=[10, 50, 100, 150, 200]
energylength = len(energies)
rows = []
#Make a list of np.arrays
for idx, i in enumerate(energies):
    spacei = idx
    tmp = pd.read_csv(f"outputpion_reco{i}_clusters.csv") #must have f to get an index in there
    for cat, row in tmp.iterrows(): #going through each event
        event = row["event"]    
        cluster_num = row["cluster_num"]
        energy = row["energy"]
        theta = row["theta"]
        phi = row["phi"]
        position = row["position"]
        num_hit = row["num_hits"]
        pos_tuple = ast.literal_eval(row["position"])
        pos_vec = np.array(pos_tuple, dtype="float")
        rows.append({
            "energy": i,
            "event": event,
            "cluster": cluster_num,
            "vector": pos_vec
        })
df = pd.DataFrame(rows)

def angle_between(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vec1 = np.linalg.norm(vector1)
    norm_vec2 = np.linalg.norm(vector2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return np.nan  # Or raise an error, depending on desired behavior
    cosine_angle = dot_product / (norm_vec1 * norm_vec2)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle_radians = np.arccos(cosine_angle)
    return angle_radians

angles = []
for (E, evt), group in df.groupby(["energy", "event"]):
    vectors = group["vector"].to_list()
    n = len(vectors)
    if n > 1:
        for i in range(n):
            for j in range(i+1, n):
                angle = angle_between(vectors[i], vectors[j])
                angles.append({
                    "energy": E,
                    "event": evt,
                    "angle_rad": angle,
                    "angle_deg": np.degrees(angle)
                })

angles_by_energy = defaultdict(list)
# fill
for angle in angles:
    E = angle["energy"]
    angles_by_energy[E].append(angle["angle_rad"])

# compute
average_angles = {}
std_angles = {}
for E, vals in angles_by_energy.items():
    vals = np.array(vals)
    average_angles[E] = np.mean(vals)
    std_angles[E] = np.std(vals, ddof=1)  # sample std

avg = [average_angles.get(E, np.nan) for E in energies ]
std = [std_angles.get(E, np.nan) for E in energies]

#Ok now for Electonrs:




elecrows = []
#Make a list of np.arrays
for idx, i in enumerate(energies):
    spacei = idx
    tmp = pd.read_csv(f"outputelectron_reco{i}_clusters.csv") #must have f to get an index in there
    for cat, row in tmp.iterrows(): #going through each event
        event = row["event"]    
        cluster_num = row["cluster_num"]
        energy = row["energy"]
        theta = row["theta"]
        phi = row["phi"]
        position = row["position"]
        num_hit = row["num_hits"]
        pos_tuple = ast.literal_eval(row["position"])
        pos_vec = np.array(pos_tuple, dtype="float")
        elecrows.append({
            "energy": i,
            "event": event,
            "cluster": cluster_num,
            "vector": pos_vec
        })
df = pd.DataFrame(elecrows)

elecangles = []
for (E, evt), group in df.groupby(["energy", "event"]):
    vectors = group["vector"].to_list()
    n = len(vectors)
    if n > 1:
        for i in range(n):
            for j in range(i+1, n):
                angle = angle_between(vectors[i], vectors[j])
                elecangles.append({
                    "energy": E,
                    "event": evt,
                    "angle_rad": angle,
                    "angle_deg": np.degrees(angle)
                })

elecangles_by_energy = defaultdict(list)
# fill
for angle in elecangles:
    E = angle["energy"]
    elecangles_by_energy[E].append(angle["angle_rad"])

# compute
elecaverage_angles = {}
elecstd_angles = {}
for E, vals in elecangles_by_energy.items():
    vals = np.array(vals)
    elecaverage_angles[E] = np.mean(vals)
    elecstd_angles[E] = np.std(vals, ddof=1)  # sample std

elecavg = [elecaverage_angles.get(E, np.nan) for E in energies ]
elecstd = [elecstd_angles.get(E, np.nan) for E in energies]






#Graphign average number of hits
#plt.errorbar(energies, avg, yerr=std, fmt='o', capsize=5, label="Average Angle between clusters for Pions")
#plt.errorbar(energies, elecavg, yerr=elecstd, fmt='o', capsize=5, label="Average Angle between clusters for Electrons")
plt.errorbar(energies, avg, fmt='o', label="Average Angle between clusters for Pions")
plt.errorbar(energies, elecavg, fmt='o', label="Average Angle between clusters for Electrons")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Angle between Clusters (Radians)")
plt.title("Cluster Angle vs. Energy")
plt.legend()
plt.grid(True)
plt.savefig("angle_vs_energy_1000.pdf")
plt.close()
