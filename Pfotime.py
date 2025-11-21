#Getting Clusters Python
#We are going to try to analyze clusters here
#I am also going to check for number of unique pdgs 
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt


#Pions
unique_pdg = []
energies=[10, 50, 100, 150, 200]
energylength = len(energies)
pfoarray = np.zeros((energylength, 1000))
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputpion_reco{i}_pfos.csv") #must have f to get an index in there
    for cat, row in df.iterrows():
        event = row["event"]
        name = row["name"]
        pfo_num = row["pfo_num"]
        energy = row["energy"]    
        momentum = row["momentum"]
        pdg = row["pdg"]
        if pdg not in unique_pdg:
            unique_pdg.append(pdg)
        charge = row["charge"]
        theta = row["theta"]
        phi = row["phi"]
        pfoarray[spacei,event] += 1
pdgarray = np.zeros((energylength, 1000, len(unique_pdg)))
for idx, i in enumerate(energies):
    df = pd.read_csv(f"outputpion_reco{i}_pfos.csv") #must have f to get an index in there
    spacei = idx
    for num, row in df.iterrows():
        event = row["event"]
        pdg = row["pdg"]
        for particle in unique_pdg:
            if pdg == particle:
                pdg_index = unique_pdg.index(pdg)
                pdgarray[spacei, event, pdg_index] += 1

mean_pdg_counts = np.mean(pdgarray, axis=1)
std_pdg_counts = np.std(pdgarray, axis=1)



#Electrons
electunique_pdg = []
energies=[10, 50, 100, 150, 200]
electenergylength = len(energies)
electpfoarray = np.zeros((electenergylength, 1000))
for idx, i in enumerate(energies):
    spacei = idx
    df = pd.read_csv(f"outputelectron_reco{i}_pfos.csv") #must have f to get an index in there
    for cat, row in df.iterrows():
        event = row["event"]
        name = row["name"]
        pfo_num = row["pfo_num"]
        energy = row["energy"]    
        momentum = row["momentum"]
        pdg = row["pdg"]
        if pdg not in electunique_pdg:
            electunique_pdg.append(pdg)
        charge = row["charge"]
        theta = row["theta"]
        phi = row["phi"]
        electpfoarray[spacei,event] += 1
electpdgarray = np.zeros((electenergylength, 1000, len(electunique_pdg)))
for idx, i in enumerate(energies):
    df = pd.read_csv(f"outputelectron_reco{i}_pfos.csv") #must have f to get an index in there
    spacei = idx
    for num, row in df.iterrows():
        event = row["event"]
        pdg = row["pdg"]
        for particle in electunique_pdg:
            if pdg == particle:
                electpdg_index = electunique_pdg.index(pdg)
                electpdgarray[spacei, event, electpdg_index] += 1

electmean_pdg_counts = np.mean(electpdgarray, axis=1)
electstd_pdg_counts = np.std(electpdgarray, axis=1)


#Other graphs:


# Define styles to cycle through
line_styles = ['-', '--', ':', '-.']
markers = ['o', 's', '^', 'D']

#Graphs 
for p_idx, pdg in enumerate(unique_pdg):
    plt.errorbar(
        energies,
        mean_pdg_counts[:, p_idx],
        #yerr=std_pdg_counts[:, p_idx],
        alpha=0.8,
        
        # --- Minimal Edits ---
        color='royalblue', # All Pions are blue
        linestyle=line_styles[p_idx % len(line_styles)], # Varied line style
        marker=markers[p_idx % len(markers)], # Varied marker
        
        label=f"Pion PDG {pdg}",
        linewidth=1.5,
        markeredgecolor='black',
        markeredgewidth=1.3
    )
for p_idx, pdg in enumerate(electunique_pdg):
    plt.errorbar(
        energies,
        electmean_pdg_counts[:, p_idx],
        #yerr=electstd_pdg_counts[:, p_idx],
        alpha=0.8,
        
        # --- Minimal Edits ---
        color='darkorange', # All Electrons are orange
        linestyle=line_styles[p_idx % len(line_styles)], # Varied line style
        marker=markers[p_idx % len(markers)], # Varied marker

        label=f" Electron PDG {pdg}",
        linewidth = 1.5,
        markeredgecolor='black',
        markeredgewidth = 1.3
    )

#Graphs 
#for p_idx, pdg in enumerate(unique_pdg):
#    plt.errorbar(
#        energies,
#        mean_pdg_counts[:, p_idx],
#        #yerr=std_pdg_counts[:, p_idx],
#        alpha=0.8,
#        fmt='-o',
#        #capsize =3,
#        label=f"Pion PDG {pdg}",
#        linewidth=1.5,
#        markeredgecolor='black',
#        markeredgewidth=1.3
#    )
#for p_idx, pdg in enumerate(electunique_pdg):
#    plt.errorbar(
#        energies,
#        electmean_pdg_counts[:, p_idx],
#        #yerr=electstd_pdg_counts[:, p_idx],
#        alpha=0.8,
#        fmt='-o',
#        #capsize = 3,
#        label=f" Electron PDG {pdg}",
#        linewidth = 1.5,
#        markeredgecolor='black',
#        markeredgewidth = 1.3
#    )
plt.xlabel("Energy (GeV)")
plt.ylabel("Mean PFO count per event")
plt.title("PFO counts by particle type vs Energy for Pions and Electrons")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pfo_counts_vs_energy_1000.pdf")
plt.close()
#How do rows work:
#Clusterarray is a 20 x 100 matrix with 20 rows as energies, 100 columns as events
#Axis = 0 means work down the rows, collapse across energies
#Axis = 1 means work across the columns, collapse across events
avg_per_energy = np.mean(pfoarray, axis=1)
std_per_energy = np.std(pfoarray, axis=1)

electavg_per_energy = np.mean(electpfoarray, axis=1)
electstd_per_energy = np.std(electpfoarray, axis=1)

#plt.errorbar(energies, avg_per_energy, yerr=std_per_energy, fmt='o-', capsize=5, label="Pfos Pion")
#plt.errorbar(energies, electavg_per_energy, yerr=electstd_per_energy, fmt='o-', capsize=5, label="Pfos Electron")
plt.errorbar(energies, avg_per_energy, fmt='o-', label="Pfos Pion")
plt.errorbar(energies, electavg_per_energy, fmt='o-', label="Pfos Electron")
plt.xlabel("Energy (GeV)")
plt.ylabel("Average Pfos per event for Pions and Electrons")
plt.title("Pfo counts vs. Energy")
plt.legend()
plt.grid(True)
plt.savefig("pfos_vs_energy_1000.pdf")
plt.close()

#Ok probably better for memory sake to only open one csv at a time


