#We are going to try to analyze clusters here
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

df = pd.read_csv("output_recoenergy3_clusters.csv")
#Ok probably better for memory sake to only open one csv at a time

CAL_COLLECTIONS = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi",
    "YokeBarrelCollection",
    "YokeEndcapCollection"
]

events = df["event"]
cluster_nums = df["cluster_num"]
energys = df["energy"]
thetas = df["theta"]
phis = df["phi"]
positions = df["position"]
num_hits = df["num_hits"]

#First I am just having the total energy in each cluster

cluster_energy = defaultdict(dict) #Prevents the need to say a key 
cluster_info = defaultdict(dict)
for idx, row in df.iterrows():
    event = row["event"]
    cluster_num = row["cluster_num"]
    energy = row["energy"]
    theta = row["theta"]
    phi = row["phi"]
    position = row["position"]
    num_hit = row["num_hits"]
    position = eval(row["position"]) #remember only use eval when you're certain it will work 
    cluster_energy[event][cluster_num] = energy #yippee this prints 
    cluster_info[event][cluster_num] = {
        "theta": theta,
        "phi": phi,
        "position": position,
        "num_hit": num_hit,
        "energy": energy,
        "event": event,
        "cluster_num": cluster_num
    }
print(cluster_info[event][cluster_num].keys())
#Ok supposedly you don't need to worry about df being open, so you can just rewrite
df = pd.read_csv("output_recoenergy3_hits.csv")
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
nested_energy = defaultdict(lambda: defaultdict(float)) #float/int fills it with zeros 

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
    systems = row["system"]
    layers = row["layer"]
    nested_energy[event][cluster_num]+=energy

#Obtaining difference in energy versus total real cluster energy
percentdif = defaultdict(dict)
totallarge = 0
for event, clusters in cluster_energy.items():
    for cluster, energy in clusters.items():
        difference = energy - nested_energy[event][cluster]
        divide = difference / energy
        total = divide * 100
        percentdif[event][cluster] = total
        if (total > 15):
            totallarge += 1
            metadata = cluster_info.get(event, {}).get(cluster, {})
            theta = metadata.get("theta")
            phi = metadata.get("phi")
            position = metadata.get("position")
            num_hit = metadata.get("num_hit")

            print(f"Event: {event}")
            print(f"Cluster: {cluster}")
            print(f"Energy: {energy}")
            print(f"Energy Difference: {total}%")
            print(f"Theta: {theta}")
            print(f"Phi: {phi}")
            print(f"Position: {position}")
            print(f"Num hits: {num_hit}")

        if (event == 4 ) and (cluster == 0):
            metadata = cluster_info.get(event, {}).get(cluster, {})
            theta = metadata.get("theta")
            phi = metadata.get("phi")
            position = metadata.get("position")
            num_hit = metadata.get("num_hit")
            print ("Working Event")
            print(f"Event: {event}")
            print(f"Cluster: {cluster}")
            print(f"Energy: {energy}")
            print(f"Energy Difference: {total}%")
            print(f"Theta: {theta}")
            print(f"Phi: {phi}")
            print(f"Position: {position}")
            print(f"Num hits: {num_hit}")




#Looking at just the clusters & events with a > 15% percent difference


#Graphing difference in energy versus total real cluster energy
x_actual_energy = []
y_difference_energy = []
for event, clusters in cluster_energy.items():
    for cluster, energy in clusters.items():
        x_actual_energy.append(energy)
        y_difference_energy.append(percentdif[event][cluster]) #should have about 200 datapoints...

plt.scatter(x_actual_energy, y_difference_energy, alpha = 0.6, s = 20 )
plt.xlabel("True Cluster Energy")
plt.ylabel("Percent Difference")
plt.title("Energy Difference vs. True Cluster Energy")
plt.grid(True)
plt.tight_layout()
#plt.savefig("energydifference3.pdf")
plt.close()

#Histogram of percent Differences
counts, bins, patches = plt.hist(y_difference_energy, bins = 50, edgecolor = 'black')
max_count_index = counts.argmax()
common_bin = (bins[max_count_index], bins[max_count_index + 1])
print(common_bin)
plt.xlabel("Percent Difference (Cluster vs Hits)")
plt.ylabel("Number of Clusters")
plt.title("Distribution of Energy Difference")
plt.tight_layout()
plt.grid(True)
#plt.savefig("percent_difference_hist3.pdf")
plt.close()

#Printing out Energy
#print("Cluster energy totals")
#for event, clusters, in cluster_energy.items():
#    for cluster, energy in clusters.items():
#        print(f"Event {event}, Cluster {cluster}: {energy}")
#print("Total hit energy per cluster")
#for event, clusters in nested_energy.items():
#    for cluster, energy in clusters.items(): #for X, Y in Z. X is a numerical number, Y is an item in list Z
#        print(f"Event {event}, Cluster {cluster}: {energy}")

#percentdif = defaultdict(dict)
#average = 0.0
#totalclusters = 0
#for event, clusters in cluster_energy.items():
#    for cluster, energy in clusters.items():
#        difference = energy - nested_energy[event][cluster]
#        divide = difference / energy 
#        total = divide * 100
#        totalclusters=totalclusters + 1
#        average = average + total
#        percentdif[event][cluster] = total


#for event, clusters in percentdif.items():
#    for cluster, dif in clusters.items():
#        print(f"Event {event}, Cluster difference {cluster}: {dif} ")

#average = average / totalclusters
#print(f"Average percent difference = {average}")