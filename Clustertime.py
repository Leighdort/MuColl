#Getting Clusters Python
#We are going to try to analyze clusters here
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

#energies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
energies=[10, 50, 100, 150, 200]
elecenergylength = len(energies)
energylength = len(energies)
elecclusterarray = np.zeros((energylength, 1000))
elechitsarray = np.zeros((energylength, 1000))
elecavghitspercluster = np.zeros((energylength, 1000))
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputelectron_reco{i}_clusters.csv") #must have f to get an index in there
    for cat, row in df.iterrows():
        event = row["event"]    
        cluster_num = row["cluster_num"]
        energy = row["energy"]
        theta = row["theta"]
        phi = row["phi"]
        position = row["position"]
        num_hit = row["num_hits"]
        position = eval(row["position"]) #remember only use eval when you're certain it will work 
        elecclusterarray[spacei,event] += 1
        elechitsarray[spacei,event] +=num_hit

x=0
y=0
while x < len(elecclusterarray):
    while y < len(elecclusterarray[0]):
        elecavghitspercluster[x][y] = elechitsarray[x][y] / elecclusterarray[x][y]
        y=y+1
    x=x+1
    y=0

elecavghits_per_energy = np.mean(elechitsarray, axis=1)
elecstdhits_perenergy = np.std(elechitsarray, axis=1)
elecavghits_per_energy_cluster = np.mean(elecavghitspercluster, axis=1)
elecstdhits_per_energy_cluster = np.std(elecavghitspercluster, axis=1)






#Now opening the pions too

pionenergylength = len(energies)
pionclusterarray = np.zeros((pionenergylength, 1000))
pionhitsarray = np.zeros((pionenergylength, 1000))
pionavghitspercluster = np.zeros((pionenergylength, 1000))
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputpion_reco{i}_clusters.csv") #must have f to get an index in there
    for cat, row in df.iterrows():
        event = row["event"]    
        cluster_num = row["cluster_num"]
        energy = row["energy"]
        theta = row["theta"]
        phi = row["phi"]
        position = row["position"]
        num_hit = row["num_hits"]
        position = eval(row["position"]) #remember only use eval when you're certain it will work 
        pionclusterarray[spacei,event] += 1
        pionhitsarray[spacei,event] +=num_hit

x=0
y=0
while x < len(pionclusterarray):
    while y < len(pionclusterarray[0]):
        pionavghitspercluster[x][y] = pionhitsarray[x][y] / pionclusterarray[x][y]
        y=y+1
    x=x+1
    y=0

pionavghits_per_energy = np.mean(pionhitsarray, axis=1)
pionstdhits_perenergy = np.std(pionhitsarray, axis=1)
pionavghits_per_energy_cluster = np.mean(pionavghitspercluster, axis=1)
pionstdhits_per_energy_cluster = np.std(pionavghitspercluster, axis=1)







#Graphing Average Number of Hits at each Energy
#Graphign average number of hits
#plt.errorbar(energies, elecavghits_per_energy, yerr=elecstdhits_perenergy, fmt='o', capsize=5, label="Average number of Hits for Electrons")
#plt.errorbar(energies, pionavghits_per_energy, yerr=pionstdhits_perenergy, fmt='o', capsize=5, label="Average number of Hits for Pions")
plt.plot(energies, elecavghits_per_energy,'o-', label="Average number of Hits for Electrons")
plt.plot(energies, pionavghits_per_energy,'o-', label="Average number of Hits for Pions")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average hits per event for Electrons/Pions")
plt.title("Hit count vs. Energy")
plt.legend()
plt.grid(True)
plt.savefig("hits_vs_energy_1000.pdf")
plt.close()

#Graphing Average Number of Hits divided by Average Number of Clusters
#Graphing average number of hits per cluster
#plt.errorbar(energies, elecavghits_per_energy_cluster, yerr=elecstdhits_per_energy_cluster, fmt='o', capsize=5, label="Average number of Hits/Cluster for Electrons" )
#plt.errorbar(energies, pionavghits_per_energy_cluster, yerr=pionstdhits_per_energy_cluster, fmt='o', capsize=5, label="Average number of Hits/Cluster for Pions" )
plt.plot(energies, elecavghits_per_energy_cluster,'o-',label="Average number of Hits/Cluster for Electrons" )
plt.plot(energies, pionavghits_per_energy_cluster,'o-', label="Average number of Hits/Cluster for Pions" )
plt.xlabel("Energy (GeV)")
plt.ylabel("Average hits per cluster for Electrons/Pions")
plt.title("Hit count per cluster vs. Energy")
plt.legend()
plt.grid(True)
plt.savefig("hitscluster_vs_energy_1000.pdf")
plt.close()

#Finding average number of Clusters

#How do rows work:
#Clusterarray is a 20 x 100 matrix with 20 rows as energies, 100 columns as events
#Axis = 0 means work down the rows, collapse across energies
#Axis = 1 means work across the columns, collapse across events
elecavg_per_energy = np.mean(elecclusterarray, axis=1)
elecstd_per_energy = np.std(elecclusterarray, axis=1)
elecmin_per_energy = np.min(elecclusterarray, axis=1)
elecmax_per_energy = np.max(elecclusterarray, axis=1)

pionavg_per_energy = np.mean(pionclusterarray, axis=1)
pionstd_per_energy = np.std(pionclusterarray, axis=1)
pionmin_per_energy = np.min(pionclusterarray, axis=1)
pionmax_per_energy = np.max(pionclusterarray, axis=1)

#Average number of Clusters per Energy
plt.plot(energies, elecavg_per_energy,'o-', label="Cluster Count Electrons")
plt.plot(energies, pionavg_per_energy,'o-', label="Cluster Count Pions")
#plt.errorbar(energies, elecavg_per_energy, yerr=elecstd_per_energy, fmt='o-', capsize=5, label="Cluster Count Electrons")
#plt.errorbar(energies, pionavg_per_energy, yerr=pionstd_per_energy, fmt='o-', capsize=5, label="Cluster Count Pions")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average clusters per event for Electrons/Pions")
plt.title("Cluster counts vs. Energy")
plt.legend()
plt.grid(True)
plt.savefig("clusters_vs_energy_1000.pdf")
plt.close()

#plt.plot(energies, avg_per_energy, 'o-', label="Mean")
#plt.plot(energies, min_per_energy, 'v--', label="Min Clusters")
#plt.plot(energies, max_per_energy, '^--', label="Max Clusters")
#plt.xlabel("Energy (GeV)")
#plt.ylabel("Clusters per event")
#plt.title("Cluster counts vs. Energy")
#plt.legend()
#plt.grid(True)
#plt.savefig("clusters_vs_energy_with_range_electrons.pdf")
#plt.close()



