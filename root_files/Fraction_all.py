#We are now going to do fraction w/ all of the clusters
#Comparing energy to total energy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

choices = [1, 2, 5, 10, 50, 100, 150, 200]
electron_mean = []
electron_low = []
electron_high = []
pion_mean = []
pion_low = []
pion_high = []

for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    events = file["events"]
    clusters = events["PandoraClusters"]
    mcparticles = events["MCParticles"]
    energies = clusters["PandoraClusters.energy"].array()
    pdg_momentum_x = mcparticles["MCParticles.momentum.x"].array()
    pdg_momentum_y = mcparticles["MCParticles.momentum.y"].array()
    pdg_momentum_z = mcparticles["MCParticles.momentum.z"].array()
    statuses = mcparticles["MCParticles.generatorStatus"].array()
    masses = mcparticles["MCParticles.mass"].array()
    print(f"Processing {num} GeV")
    fraction_for_this_energy = []
    for i in range(events.num_entries):
        status = statuses[i]
        energy = energies[i]
        mass = masses[i]
        momentum_x = pdg_momentum_x[i]
        momentum_y = pdg_momentum_y[i]
        momentum_z = pdg_momentum_z[i]
        index_particle = np.where(status == 1 )[0]
        if (len(index_particle) != 1):
            print("meow")
        if len(energy) == 0:
            #print(f"dog pion | {num} GeV | event {i} | reason: no clusters")
            continue
        # Skip if we don't have exactly one status==1 particle
        if len(index_particle) != 1:
            #print(f"cat pion | {num} GeV | event {i} | reason: found {len(index_particle)} status==1 particles")
            continue
        # Skip if any momentum or mass array is empty
        if len(mass) == 0 or len(momentum_x) == 0 or len(momentum_y) == 0 or len(momentum_z) == 0:
            #print(f"mouse pion | {num} GeV | event {i} | reason: missing MC momentum/mass")
            continue
        mx = momentum_x[index_particle]
        my = momentum_y[index_particle]
        mz = momentum_z[index_particle]
        m = mass[index_particle]
        cluster_energy = np.sum(energy)
        momentum = np.sqrt(mx**2 + my**2 + mz**2)
        mc_energy = np.sqrt(m*m + momentum*momentum)
        if cluster_energy is None or mc_energy is None:
            print("electron")
            print(num)
            print(f"event {i}")
            continue
        fraction = cluster_energy / mc_energy
        fraction_for_this_energy.append(fraction)
    fraction = np.array(fraction_for_this_energy)
    average = np.median(fraction)
    stdev = np.std(fraction)
    electron_mean.append(average)
    q16, q84 = np.percentile(fraction, [16,84])
    electron_low.append(average-q16)
    electron_high.append(q84-average)
    bins = np.linspace(np.min(fraction), np.max(fraction), 30)
    plt.hist(fraction, bins=bins, edgecolor='black')
    plt.xlabel("Ratio of all Clusters to MC Energy")
    plt.ylabel("Count")
    plt.title(f"Ratio for {num} energy electrons")
    plt.axvline(average,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {average:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"all_ratio_electrons{num}GeV.pdf")
    plt.close()

#Now we do pions
for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    events = file["events"]
    clusters = events["PandoraClusters"]
    mcparticles = events["MCParticles"]
    energies = clusters["PandoraClusters.energy"].array()
    pdg_momentum_x = mcparticles["MCParticles.momentum.x"].array()
    pdg_momentum_y = mcparticles["MCParticles.momentum.y"].array()
    pdg_momentum_z = mcparticles["MCParticles.momentum.z"].array()
    statuses = mcparticles["MCParticles.generatorStatus"].array()
    masses = mcparticles["MCParticles.mass"].array()
    print(f"Processing {num} GeV")
    fraction_for_this_energy = []
    for i in range(events.num_entries):
        status = statuses[i]
        energy = energies[i]
        mass = masses[i]
        momentum_x = pdg_momentum_x[i]
        momentum_y = pdg_momentum_y[i]
        momentum_z = pdg_momentum_z[i]
        index_particle = np.where(status == 1 )[0]
        if (len(index_particle) != 1):
            print("meow")
        if len(energy) == 0:
            #print(f"dog pion | {num} GeV | event {i} | reason: no clusters")
            continue
        # Skip if we don't have exactly one status==1 particle
        if len(index_particle) != 1:
            #print(f"cat pion | {num} GeV | event {i} | reason: found {len(index_particle)} status==1 particles")
            continue
        # Skip if any momentum or mass array is empty
        if len(mass) == 0 or len(momentum_x) == 0 or len(momentum_y) == 0 or len(momentum_z) == 0:
            #print(f"mouse pion | {num} GeV | event {i} | reason: missing MC momentum/mass")
            continue
        mx = momentum_x[index_particle]
        my = momentum_y[index_particle]
        mz = momentum_z[index_particle]
        m = mass[index_particle]
        cluster_energy = np.sum(energy)
        momentum = np.sqrt(mx**2 + my**2 + mz**2)
        mc_energy = np.sqrt(m*m + momentum*momentum)
        if cluster_energy is None or mc_energy is None:
            print("electron")
            print(num)
            print(f"event {i}")
            continue
        fraction = cluster_energy / mc_energy
        fraction_for_this_energy.append(fraction)
    fraction = np.array(fraction_for_this_energy)
    average = np.median(fraction)
    q16, q84 = np.percentile(fraction, [16, 84])
    pion_low.append(average-q16)
    pion_high.append(q84-average)
    pion_mean.append(average)
    bins = np.linspace(np.min(fraction), np.max(fraction), 30)
    plt.hist(fraction, bins=bins, edgecolor='black')
    plt.xlabel("Ratio of all Clusters to MC Energy")
    plt.ylabel("Count")
    plt.title(f"Ratio for {num} energy Pions")
    plt.axvline(average,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {average:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"all_ratio_pions{num}GeV.pdf")
    plt.close()


plt.errorbar(choices, electron_mean, yerr=[electron_low, electron_high], fmt='o', alpha=0.6, capsize=4, label="Electrons")
plt.errorbar(choices, pion_mean, yerr=[pion_low, pion_high], fmt='s', capsize=4, alpha=0.6, label="Pions")
plt.xlabel("Beam Energy")
plt.ylabel("Cluster/MC Particle")
plt.title("Cluster/MC Particle per energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_fraction_all.pdf")
plt.close()