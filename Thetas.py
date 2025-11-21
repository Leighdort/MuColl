#This is a new thingy 
#Let's first check electrons

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
elecenergylength = len(energies)
energylength = len(energies)
clusterarray = np.zeros((energylength, 1000))
pfoarray = np.zeros((energylength, 1000))
trackarray = np.zeros((energylength, 1000))
elecavghitspercluster = np.zeros((energylength, 1000))
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputpion_reco{i}_clusters.csv") #must have f to get an index in there
    print(f"reading file cluster {i}")
    for cat, row in df.iterrows():
        event = row["event"]
        print(event)    
        theta = row["theta"]
        clusterarray[idx, event] = theta
    df = pd.read_csv(f"outputpion_reco{i}_pfos.csv")
    print(f"reading file pfo {i}")
    for cat, row in df.iterrows():
        event = row["event"]
        print(event)
        theta = row["theta"]
        pfoarray[idx, event] = theta
    df = pd.read_csv(f"outputpion_reco{i}_tracks.csv")
    print(f"reading file track {i}")
    for cat, row in df.iterrows():
        print(event)
        event = row["event"]
        theta = row["theta"]
        trackarray[idx, event] = theta

averagecluster = np.mean(clusterarray, axis = 1)
stdcluster = np.std(clusterarray, axis = 1)

averagepfo = np.mean(pfoarray, axis=1)
stdpfo = np.std(pfoarray, axis=1)

averagetrack = np.mean(trackarray, axis=1)
stdtrack = np.std(trackarray, axis=1)


plt.errorbar(energies, averagecluster, yerr=stdcluster, fmt='o-', capsize=5, label="Pions Cluster")
plt.errorbar(energies, averagepfo, yerr=stdpfo, fmt='o-', capsize=5, label="Pions Pfo")
plt.errorbar(energies, averagetrack, yerr=stdtrack, fmt='o-', capsize=5, label="Pions Track")
plt.axhline(y=0.2617, color='r', linestyle='--', label='Horizontal Line at 15 degrees')
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Theta per event for Pionss")
plt.title("Theta for Pionss")
plt.legend()
plt.grid(True)
plt.savefig("pionthetas_1000.pdf")
plt.close()