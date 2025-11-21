#Missing Energy

import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

average_percent_per_energy = []
stdev_per_energy = []
#PUT EVERYTHING IN A GIANT LOOP
CAL_COLLECTIONS = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi",
    "YokeBarrelCollection",
    "YokeEndcapCollection"
]
energies = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
for idx, i in enumerate(energies):
    print(f"energy {i} started")
    df = pd.read_csv(f"outputpion_reco{i}_clusters.csv")

    cluster_energy = defaultdict(dict)
    for cat, row in df.iterrows():
        event = row["event"]
        cluster_num = row["cluster_num"]
        energy = row["energy"]
        cluster_energy[event][cluster_num] = energy

    df = pd.read_csv(f"outputpion_reco{i}_hits.csv")
    nested_energy = defaultdict(lambda: defaultdict(float))
    #Open hits
    for num, row in df.iterrows():
        event = row["event"]
        cluster_num = row["cluster_num"]
        hit_num = row["hit_num"]
        energy = row["energy"]
        nested_energy[event][cluster_num] +=energy

    percentdif = defaultdict(dict)
    for event, clusters in cluster_energy.items():
        for cluster, energy in clusters.items():
            difference = energy - nested_energy[event][cluster]
            divide = difference / energy
            total = divide * 100
            percentdif[event][cluster] = total

    list_of_percent = []
    #now we need to find average_per_energy
    for event, clusters in percentdif.items():
        for cluster, percent in clusters.items():
            list_of_percent.append(percent)
    array_percent = np.array(list_of_percent)
    average_value = np.mean(array_percent)
    std_value = np.std(array_percent)
    #Getting one value per energy
    average_percent_per_energy.append(average_value)
    stdev_per_energy.append(std_value)  

#Convert to np
average_percent_per_energy = np.array(average_percent_per_energy)
stdev_per_energy = np.array(stdev_per_energy)

plt.errorbar(energies, average_percent_per_energy, yerr=stdev_per_energy, fmt='o', capsize=5, label="Percent of Energy Loss")
plt.xlabel("Energy (GeV)")
plt.ylabel("Percent Difference of Nested energy from Cluster Energy")
plt.title("Percent energy loss as a function of Energy")
plt.yticks(np.arange(0,40,5))
plt.legend()
plt.grid(True)
plt.savefig("loss_vs_energy_pions.pdf")
plt.close()