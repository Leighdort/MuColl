#Graphing Hits Energy
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

df = pd.read_csv("output_recoenergy_hits.csv")
events = df["event"] #this gives you the full column 
cluster_nums = df["cluster_num"]
hit_nums = df["hit_num"]
x = df["x"]
y = df["y"]
z = df["z"]
energy = df["energy"]
time = df["time"]
cell_ids = df["cell_id"]
systems = df["system"]
layers = df["layer"]

#filling a nested dictionary with zeros
#nested_energy = defaultdict(lambda: defaultdict(float)) #float/int fills it with zeros 

#Looking at adding all the energy up across 99 events, cluster 0 
totalecal_energy_perlayer = defaultdict(float)
totalecal_hitsperlayer = defaultdict(float)


ecal_energy_perlayer = defaultdict(float)
ecal_hitsperlayer = defaultdict(float)
ecalend_energy_perlayer = defaultdict(float)
ecalend_hitsperlayer = defaultdict(float)
hcal_energy_perlayer = defaultdict(float)
hcal_hitsperlayer = defaultdict(float)
hcalend_energy_perlayer = defaultdict(float)
hcalend_hitsperlayer = defaultdict(float)
yoke_energy_perlayer = defaultdict(float)
yokeend_energy_perlayer = defaultdict(float)
yoke_hitsperlayer = defaultdict(float)
yokeend_hitsperlayer = defaultdict(float)
unknown_hitcount = 0
for idx, row in df.iterrows():
    event = row["event"]
    cluster_num = row["cluster_num"]
    hit_num = row["hit_num"]
    x = row["x"]
    y = row["y"]
    z = row["z"]
    energy = row["energy"]
    time = row["time"]
    cell_ids = row["cell_id"]
    systems = systems = row["system"].lower().strip().replace(",", "")
    layers = row["layer"] #layers = 0 if it's weird 
    #if cluster_num == 7 and systems == "ecal barrel":
    #    totalecal_energy_perlayer[layers] += energy
    #    totalecal_hitsperlayer[layers] += 1
    if event == 9 and cluster_num == 0:
        if layers == 0:
            unknown_hitcount += 1
        if 100000 <= cell_ids <= 999999:
            continue
        else:
            unknown_hitcount += 1
        if systems == "ecal barrel":
            ecal_energy_perlayer[layers] = ecal_energy_perlayer[layers] + energy
            ecal_hitsperlayer[layers] = ecal_hitsperlayer[layers] + 1
        if systems == "ecal endcap":
            ecalend_energy_perlayer[layers] += energy
            ecalend_hitsperlayer[layers] += 1
        if systems == "hcal barrel":
            hcal_energy_perlayer[layers] += energy
            hcal_hitsperlayer[layers] +=1
        if systems == "hcal endcap":
            hcalend_energy_perlayer[layers] += energy
            hcalend_hitsperlayer[layers] +=1
        if systems == "yoke barrel":
            yoke_energy_perlayer[layers] += energy
            yoke_hitsperlayer[layers] +=1
        if systems == "yoke endcap":
            yokeend_energy_perlayer[layers] +=energy
            yokeend_hitsperlayer[layers] +=1

print(unknown_hitcount)
#print(totalecal_hitsperlayer) 
#{0: 5.0, 1: 15.0, 2: 18.0, 3: 23.0, 4: 37.0, 5: 43.0, 6: 54.0, 7: 69.0, 8: 60.0, 9: 80.0, 10: 76.0, 11: 82.0, 12: 88.0, 13: 96.0, 14: 90.0, 15: 68.0, 16: 80.0, 17: 69.0, 18: 70.0, 19: 71.0, 20: 59.0, 21: 46.0, 22: 35.0, 23: 30.0, 24: 37.0, 25: 20.0, 26: 19.0, 27: 11.0, 28: 10.0, 29: 14.0, 30: 2.0, 31: 8.0, 32: 13.0, 33: 5.0, 34: 5.0, 37: 3.0, 39: 1.0, 38: 2.0, 36: 1.0, 35: 1.0})
#print(totalecal_energy_perlayer)
#print(yoke_energy_perlayer)
#Ok yippee this works and seems to fill them in with stuff

#Graphing total energy per cell for cluster 0 accross all 100 events in ecal Barrel 
#Graphing total number of hits for cluster 0 accross all 100 events in ecal Barrel 
#totalecal_energy_perlayer
#totalecal_hitsperlayer

#layers_to_plot = sorted(totalecal_energy_perlayer.keys())
#energies = [totalecal_energy_perlayer[layer] for layer in layers_to_plot]

layers_to_plot = sorted(yokeend_energy_perlayer.keys())
energies = [yokeend_energy_perlayer[layer] for layer in layers_to_plot]
#plt.figure(figsize=(10,6))
#plt.plot(layers_to_plot, energies, marker='o')
#plt.xlabel("Layer")
#plt.ylabel("Energy")
#plt.title("Total Ecal Barrel Energy per Layer for 100 events, Cluster 7")
#plt.grid(True, axis='y')

#plt.tight_layout()
#plt.savefig("ecalbarrel_allevents_cluster7_more.pdf")
#Let's start with graphing
#Let's just graph ecal barrel of event 0, cluster_num 0
#layers_to_plot = sorted(hcal_energy_perlayer.keys()) #getting keys means the layer number and sorted in ascending order
#energies = [hcal_energy_perlayer[layer] for layer in layers_to_plot] #loops through layers and calls the energies that correspond to them 

#plt.figure(figsize=(10,6)) #wait I can probably find uncertainties in the energy 
plt.plot(layers_to_plot, energies, marker='o')
plt.xlabel("Layer")
plt.ylabel("Energy")
plt.title("Yoke End Energy per Layer for Event 9, Cluster 0")
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig("yokeend_event9_cluster0_energy.pdf")
plt.close()

#Now check all 99 events or all clusters -> total energy over X events and clusters per part idk 
#Average energy per layer per cluster? 

#Summate the hits in ecal for 120 events
#Do clusters 1 look different than 2 than 3 than idk 