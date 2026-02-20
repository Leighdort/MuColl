
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot


#First we will get mean and stdev
#And we will graph log scale
#Then we will pick some events, deal
choices = [1, 2, 5, 10, 50, 100, 150, 200]

#Now we resume what we need
electron_10 = []
electron_20 = []
electron_30 = []
pion_10 = []
pion_20 = []
pion_30 = []
e_ten_ratio = []
e_twenty_ratio = []
e_thirty_ratio = []
e_10_low = []
e_10_high = []
e_20_low = []
e_20_high = []
e_30_low = []
e_30_high = []
p_ten_ratio = []
p_twenty_ratio = []
p_thirty_ratio = []
p_10_low = []
p_10_high = []
p_20_low = []
p_20_high = []
p_30_low = []
p_30_high = []

for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]
    print(num)
    clusters = events["PandoraClusters"]
    cluster_hit_begin = clusters["PandoraClusters.clusters_begin"].array()
    mcparticles = events["MCParticles"]
    energies = clusters["PandoraClusters.energy"].array()
    pdg_momentum_x = mcparticles["MCParticles.momentum.x"].array()
    pdg_momentum_y = mcparticles["MCParticles.momentum.y"].array()
    pdg_momentum_z = mcparticles["MCParticles.momentum.z"].array()
    statuses = mcparticles["MCParticles.generatorStatus"].array()
    masses = mcparticles["MCParticles.mass"].array()
    ten = 0
    twent5 = 0
    thirty = 0
    e_10 = []
    e_20 = []
    e_30 = []
    for i in range(events.num_entries):
        energy1 = energies[i]
        if i % 1000 == 0:
            print(i)
        length = len(cluster_hit_begin[i])
        fraction = 0 
        if length >= 10:
            status = statuses[i]
            mass = masses[i]
            momentum_x = pdg_momentum_x[i]
            momentum_y = pdg_momentum_y[i]
            momentum_z = pdg_momentum_z[i]
            index_particle = np.where(status == 1)[0]
            if len(energy1) == 0:
                #no clsuters
                continue
            if len(index_particle) != 1:
                continue
                #too many og particles
            if len(mass) == 0 or len(momentum_x) == 0 or len(momentum_y) == 0 or len(momentum_z) == 0:
                #no momentum or mass
                continue
            mx = momentum_x[index_particle]
            my = momentum_y[index_particle]
            mz = momentum_z[index_particle]
            m = mass[index_particle]
            cluster_energy = np.sum(energies[i])
            momentum = np.sqrt(mx**2 + my**2 + mz**2)
            mc_energy = np.sqrt(m*m + momentum*momentum)
            fraction = cluster_energy / mc_energy
            e_10.append(fraction)
            ten += 1
        if length >= 20:
            twent5 += 1
            e_20.append(fraction)
        if length >= 30:
            thirty += 1
            e_30.append(fraction)
    electron_10.append(ten)
    electron_20.append(twent5)
    electron_30.append(thirty)
    e_10 = np.array(e_10)
    e_20 = np.array(e_20)
    e_30 = np.array(e_30)
    if len(e_10) == 0:
        e_ten_ratio.append(np.nan)
        e_10_low.append(0)
        e_10_high.append(0)
    else:
        median_10 = np.median(e_10)
        q16, q84 = np.percentile(e_10, [16,84])
        e_10_low.append(median_10-q16)
        e_10_high.append(q84-median_10)
        e_ten_ratio.append(median_10)
    if len(e_20) == 0:
        e_twenty_ratio.append(np.nan)
        e_20_low.append(0)
        e_20_high.append(0)
    else:
        median_20 = np.median(e_20)
        q16, q84 = np.percentile(e_20, [16,84])
        e_20_low.append(median_20-q16)
        e_20_high.append(q84-median_20)
        e_twenty_ratio.append(median_20)
    if len(e_30) == 0:
        e_thirty_ratio.append(np.nan)
        e_30_low.append(0)
        e_30_high.append(0)
    else:
        median_30 = np.median(e_30)
        q16, q84 = np.percentile(e_30, [16,84])
        e_30_low.append(median_30-q16)
        e_30_high.append(q84-median_30)
        e_thirty_ratio.append(median_30)
#Now I beleive margins are correct for electrons, now must do for pions

for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    print(num)
    clusters = events["PandoraClusters"]
    cluster_hit_begin = clusters["PandoraClusters.clusters_begin"].array()
    ten = 0
    twent5 = 0
    thirty = 0
    mcparticles = events["MCParticles"]
    energies = clusters["PandoraClusters.energy"].array()
    pdg_momentum_x = mcparticles["MCParticles.momentum.x"].array()
    pdg_momentum_y = mcparticles["MCParticles.momentum.y"].array()
    pdg_momentum_z = mcparticles["MCParticles.momentum.z"].array()
    statuses = mcparticles["MCParticles.generatorStatus"].array()
    masses = mcparticles["MCParticles.mass"].array()
    e_10 = []
    e_20 = []
    e_30 = []
    for i in range(events.num_entries):
        energy1=energies[i]
        if i % 1000 == 0:
            print(i)
        length = len(cluster_hit_begin[i])
        fraction = 0
        if length >= 10:
            ten += 1
            status = statuses[i]
            mass = masses[i]
            momentum_x = pdg_momentum_x[i]
            momentum_y = pdg_momentum_y[i]
            momentum_z = pdg_momentum_z[i]
            index_particle = np.where(status == 1)[0]
            if len(energy1) == 0:
                #no clsuters
                continue
            if len(index_particle) != 1:
                continue
                #too many og particles
            if len(mass) == 0 or len(momentum_x) == 0 or len(momentum_y) == 0 or len(momentum_z) == 0:
                #no momentum or mass
                continue
            mx = momentum_x[index_particle]
            my = momentum_y[index_particle]
            mz = momentum_z[index_particle]
            m = mass[index_particle]
            cluster_energy = np.sum(energies[i])
            momentum = np.sqrt(mx**2 + my**2 + mz**2)
            mc_energy = np.sqrt(m*m + momentum*momentum)
            fraction = cluster_energy / mc_energy
            e_10.append(fraction)
        if length >= 20:
            twent5 += 1
            e_20.append(fraction)
        if length >= 30:
            thirty += 1
            e_30.append(fraction)
    pion_10.append(ten)
    pion_20.append(twent5)
    pion_30.append(thirty)
    e_10 = np.array(e_10)
    e_20 = np.array(e_20)
    e_30 = np.array(e_30)
    if len(e_10) == 0:
        p_ten_ratio.append(np.nan)
        p_10_low.append(0)
        p_10_high.append(0)
    else:
        median_10 = np.median(e_10)
        q16, q84 = np.percentile(e_10, [16,84])
        p_10_low.append(median_10-q16)
        p_10_high.append(q84-median_10)
        p_ten_ratio.append(median_10)
    if len(e_20) == 0:
        p_twenty_ratio.append(np.nan)
        p_20_low.append(0)
        p_20_high.append(0)
    else:
        median_20 = np.median(e_20)
        q16, q84 = np.percentile(e_20, [16,84])
        p_20_low.append(median_20-q16)
        p_20_high.append(q84-median_20)
        p_twenty_ratio.append(median_20)
    if len(e_30) == 0:
        p_thirty_ratio.append(np.nan)
        p_30_low.append(0)
        p_30_high.append(0)
    else:
        median_30 = np.median(e_30)
        q16, q84 = np.percentile(e_30, [16,84])
        p_30_low.append(median_30-q16)
        p_30_high.append(q84-median_30)
        p_thirty_ratio.append(median_30)


#Plotting the Normal Cluster count
plt.plot(choices, pion_10, 's', color='tab:blue', alpha=0.6, label="Pion, 10+ clusters")
plt.plot(choices, pion_20, '^', color='tab:blue', alpha=0.6, label="Pion, 20+ clusters")
plt.plot(choices, pion_30, 'o', color='tab:blue', alpha=0.6, label="Pion, 30+ clusters")
plt.plot(choices, electron_10, 's', color='tab:orange', alpha=0.6, label="Electron, 10+ clusters")
plt.plot(choices, electron_20, '^', color='tab:orange', alpha=0.6, label="Electron, 20+ clusters")
plt.plot(choices, electron_30, 'o', color='tab:orange', alpha=0.6, label="Electron, 30+ clusters")
plt.xlabel("Transverse Beam Energy")
plt.ylabel("Number of Events over Cluster Threshold")
plt.title("Cluster Thresholds out of 10,000 Events")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cluster_threshhold.pdf")
plt.close()

#Now we are plotting the MC Energy
plt.errorbar(choices, p_ten_ratio, yerr = [p_10_low, p_10_high], fmt='s', color='tab:blue', alpha=0.6, label="Pion, 10+ clusters")
plt.errorbar(choices, p_twenty_ratio, yerr = [p_20_low, p_20_high], fmt='^', color='tab:blue', alpha=0.6, label="Pion, 20+ clusters")
plt.errorbar(choices, p_thirty_ratio, yerr = [p_30_low, p_30_high], fmt='o', color='tab:blue', alpha=0.6, label="Pion, 30+ clusters")
plt.errorbar(choices, e_ten_ratio, yerr = [e_10_low, e_10_high], fmt='s', color='tab:orange', alpha=0.6, label="Electron, 10+ clusters")
plt.errorbar(choices, e_twenty_ratio, yerr = [e_20_low, e_20_high], fmt='^', color='tab:orange', alpha=0.6, label="Electron, 20+ clusters")
plt.errorbar(choices, e_thirty_ratio, yerr = [e_30_low, e_30_high], fmt='o', color='tab:orange', alpha=0.6, label="Electron, 30+ clusters")
plt.xlabel("Transverse Beam Energy")
plt.ylabel("Select Events Cluster/MC Particle")
plt.title("High Cluster Events ratio for Cluster/MC Particle")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_all_highclust.pdf")
plt.close()

