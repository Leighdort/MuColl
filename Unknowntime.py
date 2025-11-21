#Unknown hits
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt


CAL_COLLECTIONS = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi",
    "YokeBarrelCollection",
    "YokeEndcapCollection"
]

energies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
energylength = len(energies)
unknownarray = np.zeros((energylength, 100))
standardunknownarray= np.zeros((energylength, 100))
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputpion_reco{i}_unknown_hits.csv") #must have f to get an index in there
    for cat, row in df.iterrows():
        event = row["event"]
        hit_num = row["hit_num"]    
        energy = row["energy"]
        cluster_num = row["cluster_num"]
        location = row["location"]
        layer = row["layer"]
        position = row["position"]
        unknownarray[spacei, event] += 1

avgunknown = np.mean(unknownarray, axis=1)
stdunknown = np.std(unknownarray, axis=1)


clusterarray = np.zeros((energylength, 100))
hitsarray = np.zeros((energylength, 100))
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputpion_reco{i}_clusters.csv") #must have f to get an index in there
    for idx, row in df.iterrows():
        event = row["event"]    
        cluster_num = row["cluster_num"]
        energy = row["energy"]
        theta = row["theta"]
        phi = row["phi"]
        position = row["position"]
        num_hit = row["num_hits"]
        position = eval(row["position"]) #remember only use eval when you're certain it will work 
        hitsarray[spacei,event] +=num_hit

x=0
y=0
while x < len(unknownarray):
    while y < len(unknownarray[0]):
        standardunknownarray[x][y] = unknownarray[x][y]/hitsarray[x][y]
        y=y+1
    x=x+1
    y=0

avgstandardarray = np.mean(standardunknownarray, axis=1)
stdstandardarray = np.std(standardunknownarray, axis=1)


#Graphing just the number of unknown hits
plt.errorbar(energies, avgunknown, yerr=stdunknown, fmt='o', capsize=5, label="Average unknown Hits")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Unknown hits per event for Pions")
plt.title("Unknown Hit count vs. Energy")
plt.legend()
plt.grid(True)
plt.savefig("unknownhits_vs_energy_pions.pdf")
plt.close()

plt.errorbar(energies, avgstandardarray, yerr=stdstandardarray, fmt='o', capsize=5, label="Unknownhits / Totalhits")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Unknown hits / Total Hits per Event for Pions")
plt.title("Unknownits/total hits vs. Energy")
plt.legend()
plt.grid(True)
plt.savefig("standardunknownhits_vs_energy_pions.pdf")

