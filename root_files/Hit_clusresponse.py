#Cluster Energy vs Hit Energy
#This is graphing hit based resolution vs cluster based resolution
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

# System IDs
ECAL_BARREL = 679272617
HCAL_BARREL = 1573202488
ECAL_ENDCAP = 3383333369
HCAL_ENDCAP = 2381985645

real_systems = [
    "EcalBarrelCollectionRec",
    "HcalBarrelCollectionRec",
    "EcalEndcapCollectionRec",
    "HcalEndcapCollectionRec"
]


clus_high = []
clus_low = []
clus_median = []
hits_high = []
hits_low = []
hits_median = []
energy = [10, 30, 50]
particle = [11, 211]
angles = [15, 85]

#angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for pid in particle:
    median_soft = {15: [], 85: []}
    low_soft    = {15: [], 85: []}
    high_soft   = {15: [], 85: []}
    median_nosoft = {15: [], 85: []}
    low_nosoft    = {15: [], 85: []}
    high_nosoft   = {15: [], 85: []}
    for e in energy: 
        for a in angles:
            difference = []
            print(f"\n=== Energy {a} Degrees ===")
            file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_basesoft/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_bib2/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            events = file["events"]

            pandora_clusters = events["PandoraClusters"]
            pandora_clusters_hits = events["_PandoraClusters_hits"]
            mcparticles = events["MCParticles"]

            # Load arrays once
            status_mc = mcparticles["MCParticles.generatorStatus"].array()

            mc_momx = mcparticles["MCParticles.momentum.x"].array()
            mc_momy = mcparticles["MCParticles.momentum.y"].array()
            mc_momz = mcparticles["MCParticles.momentum.z"].array()
            mc_mass = mcparticles["MCParticles.mass"].array()

            cluster_energy = pandora_clusters["PandoraClusters.energy"].array()

            hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
            hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

            collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
            hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()

            # Preload energy maps
            energy_map = {}
            for name in real_systems:
                prefix = f"{name}/{name}"
                energy_map[name] = events[f"{prefix}.energy"].array()

            for i in range(5000):

                if len(cluster_energy[i]) != 1:
                    continue

                mask = (status_mc[i] == 1)
                if not np.any(mask):
                    continue
                oneclusenergy = cluster_energy[i]

                hits_begin_arr = hits_begin_all[i]
                hits_end_arr   = hits_end_all[i]
                collection_ID  = collectionID_all[i]
                hit_index      = hit_index_all[i]

                EOE = 0.0

                for j in range(len(hits_begin_arr)):
                    lo = hits_begin_arr[j]
                    hi = hits_end_arr[j]

                    sysIDs = collection_ID[lo:hi]
                    idxs   = hit_index[lo:hi]

                    # 🔥 vectorized masks
                    mask_eb = (sysIDs == ECAL_BARREL)
                    mask_hb = (sysIDs == HCAL_BARREL)
                    mask_ee = (sysIDs == ECAL_ENDCAP)
                    mask_he = (sysIDs == HCAL_ENDCAP)

                    # 🔥 vectorized energy sums
                    if np.any(mask_eb):
                        EOE += np.sum(energy_map["EcalBarrelCollectionRec"][i][idxs[mask_eb]])
                    if np.any(mask_hb):
                        EOE += np.sum(energy_map["HcalBarrelCollectionRec"][i][idxs[mask_hb]])
                    if np.any(mask_ee):
                        EOE += np.sum(energy_map["EcalEndcapCollectionRec"][i][idxs[mask_ee]])
                    if np.any(mask_he):
                        EOE += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])
                #Let's subtract hit energy - cluster energy / cluster energy
                percent_dif = ((oneclusenergy - EOE) / oneclusenergy)
                difference.append(percent_dif)

            difference = np.array(difference)
            median = np.median(difference)
            q16, q84 = np.percentile(difference, [16, 84])
            median_nosoft[a].append(median)
            low_nosoft[a].append(median - q16)
            high_nosoft[a].append(q84 - median)
            #The following is for histogram making, is currently turned off 
            '''
            bins = 60
            plt.figure(figsize=(7,5))
            plt.hist(difference, bins=bins, edgecolor = 'black')
            plt.axvline(
                median,
                linestyle='--',
                linewidth=2,
                label=f'Median = {median:.3f}'
            )
            plt.xlabel("Difference in Cluster Energy to Hit Energy")
            plt.ylabel("Count")
            plt.title(f" Difference in Energies ({pid}, {a}°, {e} GeV) No Software Nobib")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"hitclus_energydifference_{a}_{pid}_{e}_no_software_nobib.pdf")
            plt.close()
            '''
            difference = []
            print(f"\n=== Energy {a} Degrees ===")
            file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_bib2/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_bib2/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            events = file["events"]

            pandora_clusters = events["PandoraClusters"]
            pandora_clusters_hits = events["_PandoraClusters_hits"]
            mcparticles = events["MCParticles"]

            # Load arrays once
            status_mc = mcparticles["MCParticles.generatorStatus"].array()

            mc_momx = mcparticles["MCParticles.momentum.x"].array()
            mc_momy = mcparticles["MCParticles.momentum.y"].array()
            mc_momz = mcparticles["MCParticles.momentum.z"].array()
            mc_mass = mcparticles["MCParticles.mass"].array()

            cluster_energy = pandora_clusters["PandoraClusters.energy"].array()

            hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
            hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

            collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
            hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()

            # Preload energy maps
            energy_map = {}
            for name in real_systems:
                prefix = f"{name}/{name}"
                energy_map[name] = events[f"{prefix}.energy"].array()

            for i in range(5000):

                if len(cluster_energy[i]) != 1:
                    continue

                mask = (status_mc[i] == 1)
                if not np.any(mask):
                    continue
                oneclusenergy = cluster_energy[i]

                hits_begin_arr = hits_begin_all[i]
                hits_end_arr   = hits_end_all[i]
                collection_ID  = collectionID_all[i]
                hit_index      = hit_index_all[i]

                EOE = 0.0

                for j in range(len(hits_begin_arr)):
                    lo = hits_begin_arr[j]
                    hi = hits_end_arr[j]

                    sysIDs = collection_ID[lo:hi]
                    idxs   = hit_index[lo:hi]

                    # 🔥 vectorized masks
                    mask_eb = (sysIDs == ECAL_BARREL)
                    mask_hb = (sysIDs == HCAL_BARREL)
                    mask_ee = (sysIDs == ECAL_ENDCAP)
                    mask_he = (sysIDs == HCAL_ENDCAP)

                    # 🔥 vectorized energy sums
                    if np.any(mask_eb):
                        EOE += np.sum(energy_map["EcalBarrelCollectionRec"][i][idxs[mask_eb]])
                    if np.any(mask_hb):
                        EOE += np.sum(energy_map["HcalBarrelCollectionRec"][i][idxs[mask_hb]])
                    if np.any(mask_ee):
                        EOE += np.sum(energy_map["EcalEndcapCollectionRec"][i][idxs[mask_ee]])
                    if np.any(mask_he):
                        EOE += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])
                #Let's subtract hit energy - cluster energy / cluster energy
                percent_dif = ((oneclusenergy - EOE) / oneclusenergy)
                difference.append(percent_dif)
            difference = np.array(difference)
            median = np.median(difference)
            q16, q84 = np.percentile(difference, [16, 84])
            median_soft[a].append(median)
            low_soft[a].append(median - q16)
            high_soft[a].append(q84 - median)
    #Here we make summery plots
    for a in angles:
        plt.errorbar(
            energy,
            median_nosoft[a],
            yerr=[low_nosoft[a], high_nosoft[a]],
            fmt='o-',
            capsize=4,
            label=f"{a}° Software Off"
        )
        plt.errorbar(
            energy,
            median_soft[a],
            yerr=[low_soft[a], high_soft[a]],
            fmt='s--',
            capsize=4,
            label=f"{a}° Software On"
        )
    plt.xlabel("Beam Energy")
    plt.ylabel("Median Difference in Cluster Energy to Hit Energy")
    plt.title(f"Fractional Difference for {pid} between Clusters and Hits")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"summery_fractionaldiff_hitclus_{pid}.pdf")
    plt.close()

