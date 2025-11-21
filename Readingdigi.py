#Here I am going to explore using a csv 
import pandas as pd 
import numpy as np
import math 

df = pd.read_csv("output_digidif_hits.csv")
#df means a dataframe that is a table

CAL_COLLECTIONS = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi",
]

#column access = df
events = df["event"]
hit_nums = df["hit_num"]
types = df["type"]
energies = df["energy"]
positions = df["position"]

EcalBhits = np.zeros(10)
EcalEhits = np.zeros(10)
HcalBhits = np.zeros(10)
HcalEhits = np.zeros(10)

EtEcalB = np.zeros(10)
EtEcalE = np.zeros(10)
EtHcalB = np.zeros(10)
EtHcalE = np.zeros(10)


#Iterate through rows
#only do df.iterrows when you know the types of information -> does not preserve tuples or whatnot
for idx, row in df.iterrows():
    event = row["event"]
    hit_num = row["hit_num"]
    hit_type = row["type"]
    energy = row["energy"]
    position = eval(row["position"]) #turns the string into a python tuple of floats
    if hit_type == "EcalBarrelCollectionDigi" :
        EcalBhits[event] = EcalBhits[event] + 1
        EtEcalB[event] = EtEcalB[event] + energy
    elif hit_type == "EcalEndcapCollectionDigi" :
        EcalEhits[event] = EcalEhits[event] + 1
        EtEcalE[event] = EtEcalE[event] + energy
    elif hit_type == "HcalBarrelCollectionDigi" :
        HcalBhits[event] = HcalBhits[event] + 1
        EtHcalB[event] = EtHcalB[event] + energy
    elif hit_type == "HcalEndcapCollectionDigi" :
        HcalEhits[event] = HcalEhits[event] + 1
        EtHcalE[event] = EtHcalE[event] + energy

#The first thing I will find is average hits per section
avgEcalBhits = 0
avgEcalEhits = 0
avgHcalBhits = 0
avgHcalEhits = 0

#Finding average energy exerted in the collector between all 10 events
avgEtEcalB = np.mean(EtEcalB)
avgEtEcalE = np.mean(EtEcalE)
avgEtHcalB = np.mean(EtHcalB)
avgEtHcalE = np.mean(EtHcalE)

#I am going to take the energy per area per event divide by the particle per area per event. then take an average of these numbers
avgperpartEcalB = np.zeros(10)
avgperpartEcalE = np.zeros(10)
avgperpartHcalB = np.zeros(10)
avgperpartHcalE = np.zeros(10)

print(f"Average total energy in the EcalBarrel {avgEtEcalB}")
print(f"Average total energy in the EcalEndcap {avgEtEcalE}")
print(f"Average total energy in the HcalBarrel {avgEtHcalB}")
print(f"Average total energy in the HcalEndcap {avgEtHcalE}")

i = 0
while i < 10:
    if EcalBhits[i] > 0:
        avgperpartEcalB[i] = EtEcalB[i] / EcalBhits[i]
    else:
        avgperpartEcalB[i] = 0  # or np.nan

    if EcalEhits[i] > 0:
        avgperpartEcalE[i] = EtEcalE[i] / EcalEhits[i]
    else:
        avgperpartEcalE[i] = 0

    if HcalBhits[i] > 0:
        avgperpartHcalB[i] = EtHcalB[i] / HcalBhits[i]
    else:
        avgperpartHcalB[i] = 0

    if HcalEhits[i] > 0:
        avgperpartHcalE[i] = EtHcalE[i] / HcalEhits[i]
    else:
        avgperpartHcalE[i] = 0
    i = i + 1

avgpartEcalB = np.mean(avgperpartEcalB)
avgpartEcalE = np.mean(avgperpartEcalE)
avgpartHcalB = np.mean(avgperpartHcalB)
avgpartHcalE = np.mean(avgperpartHcalE)

print(f"Average energy per hit in the EcalBarrel {avgpartEcalB}")
print(f"Average energy per hit in the EcalEndcap {avgpartEcalE}")
print(f"Average energy per hit in the HcalBarrel {avgpartHcalB}")
print(f"Average energy per hit in the HcalEndcap {avgpartHcalE}")

i = 0

#Instead of doing this, should have done np.mean(EcalBhits)
while i < 10:
    avgEcalBhits = avgEcalBhits + EcalBhits[i]
    avgEcalEhits = avgEcalEhits + EcalEhits[i]
    avgHcalBhits = avgHcalBhits + HcalBhits[i]
    avgHcalEhits = avgHcalEhits + HcalEhits[i]
    i=i+1

avgEcalBhits = avgEcalBhits/10.0
avgEcalEhits = avgEcalEhits/10.0
avgHcalBhits = avgHcalBhits/10.0
avgHcalEhits = avgHcalEhits/10.0

print(f"Average number of hits to the EcalBarrel is {avgEcalBhits} ")
print(f"Average number of hits to the EcalEndcap is {avgEcalEhits} ")
print(f"Average number of hits to the HcalBarrel is {avgHcalBhits} ")
print(f"Average number of hits to the HcalEndcap is {avgHcalEhits} ")



