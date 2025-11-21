#Getting Clusters Python
#We are going to try to analyze clusters here
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt



#energies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
energies=[10, 50, 100, 150, 200]
energylength = len(energies)
trackarray = np.zeros((energylength, 1000))
print("Pions")
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputpion_reco{i}_tracks.csv") #must have f to get an index in there
    for cat, row in df.iterrows():
        event = row["event"]
        name = row["name"]
        p = row["p"]
        pt = row["pt"]
        pz = row["pz"]
        theta = row["theta"]
        phi = row["phi"]
        chi2_ndf = row["chi2/ndf"]
        trackarray[spacei,event] += 1
    for meow, rowy in df.iterrows():
        event = rowy["event"]
        if trackarray[spacei, event] !=1: 
            print (f"energy: {i}, event: {event}, tracknum = {trackarray[spacei, event]}")


#How do rows work:
#Clusterarray is a 20 x 100 matrix with 20 rows as energies, 100 columns as events
#Axis = 0 means work down the rows, collapse across energies
#Axis = 1 means work across the columns, collapse across events
avg_per_energy = np.mean(trackarray, axis=1)
std_per_energy = np.std(trackarray, axis=1)

print("electrons")
electrackarray = np.zeros((energylength, 1000))
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputelectron_reco{i}_tracks.csv") #must have f to get an index in there
    for cat, row in df.iterrows():
        event = row["event"]
        name = row["name"]
        p = row["p"]
        pt = row["pt"]
        pz = row["pz"]
        theta = row["theta"]
        phi = row["phi"]
        chi2_ndf = row["chi2/ndf"]
        electrackarray[spacei,event] += 1
    for meow, rowy in df.iterrows():
        event = rowy["event"]
        if electrackarray[spacei, event] !=1: 
            print (f"energy: {i}, event: {event}, tracknum = electrackarray{[spacei, event]}")
elecavg_per_energy = np.mean(electrackarray, axis = 1)
elecstd_per_energy = np.std(electrackarray, axis = 1)


plt.errorbar(energies, avg_per_energy, yerr=std_per_energy, fmt='o-', capsize=5, label="Pions Pfos")
plt.errorbar(energies, elecavg_per_energy, yerr=elecstd_per_energy, fmt='o-', capsize=5, label="Electrons Pfos")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Tracks per event for Pions")
plt.title("Track counts vs. Energy")
plt.legend()
plt.grid(True)
plt.savefig("tracks_vs_energy_1000.pdf")
plt.close()


#Ok probably better for memory sake to only open one csv at a time


