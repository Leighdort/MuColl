#Finding the hits in each layer of the detector
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
    "YokeEndcapCollection",
    "Unknown"
]

energies=[10, 50, 100, 150, 200]
#energies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
hitlocation = np.zeros((len(energies), 1000, 7))
for idx, i in enumerate(energies):
    print(idx)
    print(i)
    spacei = idx
    df = pd.read_csv(f"outputpion_reco{i}_hits.csv")
    for x, row in df.iterrows():
        event = row["event"]
        systems = row["system"].lower().strip().replace(",", "")
        if systems == "ecal barrel":
            hitlocation[spacei, event, 0] += 1
        elif systems == "ecal endcap":
            hitlocation[spacei, event, 1] += 1
        elif systems == "hcal barrel":
            hitlocation[spacei, event, 2] += 1
        elif systems == "hcal endcap":
            hitlocation[spacei, event, 3] += 1
        elif systems == "yoke barrel":
            hitlocation[spacei, event, 4] += 1
        elif systems == "yoke endcap":
            hitlocation[spacei, event, 5] += 1
        else:
            hitlocation[spacei, event, 6] += 1

mean_hit_counts = np.mean(hitlocation, axis=1)
std_hit_counts = np.std(hitlocation, axis=1)

elechitlocation = np.zeros((len(energies), 1000, 7))
for idx, i in enumerate(energies):
    print(idx)
    print(i)
    spacei = idx
    df = pd.read_csv(f"outputelectron_reco{i}_hits.csv")
    for x, row in df.iterrows():
        event = row["event"]
        systems = row["system"].lower().strip().replace(",", "")
        if systems == "ecal barrel":
            elechitlocation[spacei, event, 0] += 1
        elif systems == "ecal endcap":
            elechitlocation[spacei, event, 1] += 1
        elif systems == "hcal barrel":
            elechitlocation[spacei, event, 2] += 1
        elif systems == "hcal endcap":
            elechitlocation[spacei, event, 3] += 1
        elif systems == "yoke barrel":
            elechitlocation[spacei, event, 4] += 1
        elif systems == "yoke endcap":
            elechitlocation[spacei, event, 5] += 1
        else:
            elechitlocation[spacei, event, 6] += 1

elecmean_hit_counts = np.mean(elechitlocation, axis=1)
elecstd_hit_counts = np.std(elechitlocation, axis=1)


from matplotlib import cm
from matplotlib.lines import Line2D
import numpy as np

line_styles = ['-', '--', ':', '-.']
markers = ['o', 's', '^', 'D']

# use a colormap with as many unique colors as collections
colors = cm.get_cmap('tab10', len(CAL_COLLECTIONS)).colors

# --- Plot Pions ---
fig, ax = plt.subplots(figsize=(8,6))

for ids, location in enumerate(CAL_COLLECTIONS):
    ax.errorbar(
        energies,
        mean_hit_counts[:, ids],
        #yerr=std_hit_counts[:, ids],
        alpha=0.8,
        color=colors[ids % len(colors)],       # <-- different color per collection
        linestyle=line_styles[ids % len(line_styles)],
        marker=markers[ids % len(markers)],
        label=location,
        linewidth=1.5,
        markeredgecolor='black',
        markeredgewidth=1.0
    )

ax.set_xlabel("Energy (GeV)")
ax.set_ylabel("Mean Hit Count")
ax.set_title("Pion Hit Counts by Location vs Energy")
ax.grid(True)
ax.legend(ncol=2, title="Detector Location")
plt.tight_layout()
plt.savefig("hit_location_vs_energy_pions.pdf")
plt.close()


# --- Plot Electrons ---
fig, ax = plt.subplots(figsize=(8,6))

for ids, location in enumerate(CAL_COLLECTIONS):
    ax.errorbar(
        energies,
        elecmean_hit_counts[:, ids],
        #yerr=elecstd_hit_counts[:, ids],
        alpha=0.8,
        color=colors[ids % len(colors)],       # <-- same scheme for consistency
        linestyle=line_styles[ids % len(line_styles)],
        marker=markers[ids % len(markers)],
        label=location,
        linewidth=1.5,
        markeredgecolor='black',
        markeredgewidth=1.0
    )

ax.set_xlabel("Energy (GeV)")
ax.set_ylabel("Mean Hit Count")
ax.set_title("Electron Hit Counts by Location vs Energy")
ax.grid(True)
ax.legend(ncol=2, title="Detector Location")
plt.tight_layout()
plt.savefig("hit_location_vs_energy_electrons.pdf")
plt.close()

