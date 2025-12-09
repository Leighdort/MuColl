#Comparing energy to total energy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

choices = [1, 2, 5, 10, 50, 100, 150, 200]
#First I will see the ratio of particle w/ status 0, its energy to leading cluster energy
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
            print(f"dog pion | {num} GeV | event {i} | reason: no clusters")
            continue
        # Skip if we don't have exactly one status==1 particle
        if len(index_particle) != 1:
            print(f"cat pion | {num} GeV | event {i} | reason: found {len(index_particle)} status==1 particles")
            continue
        # Skip if any momentum or mass array is empty
        if len(mass) == 0 or len(momentum_x) == 0 or len(momentum_y) == 0 or len(momentum_z) == 0:
            print(f"mouse pion | {num} GeV | event {i} | reason: missing MC momentum/mass")
            continue
        mx = momentum_x[index_particle]
        my = momentum_y[index_particle]
        mz = momentum_z[index_particle]
        m = mass[index_particle]
        cluster_energy = np.max(energy)
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
    bins = np.linspace(np.min(fraction), np.max(fraction), 30)
    plt.hist(fraction, bins=bins, edgecolor='black')
    plt.xlabel("Ratio of Leading Cluster Energy to MC Energy")
    plt.ylabel("Count")
    plt.title(f"Ratio for {num} energy pions")
    plt.tight_layout()
    plt.savefig(f"ratio_pions{num}GeV.pdf")
    plt.close()




