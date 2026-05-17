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
particle = [11]
angles = [15, 85, 140]
#angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for pid in particle:
    for a in angles:
        fraction_hits = []
        fraction_clus = []
        print(f"\n=== Energy {a} Degrees ===")

        file = uproot.open(
            f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial6/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root"
        )
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

        for i in range(events.num_entries):

            if len(cluster_energy[i]) != 1:
                continue

            mask = (status_mc[i] == 1)
            if not np.any(mask):
                continue

            momx = mc_momx[i][mask][0]
            momy = mc_momy[i][mask][0]
            momz = mc_momz[i][mask][0]
            mcmass = mc_mass[i][mask][0]

            mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
            mc_energy = np.sqrt(mcmass**2 + mc_momentum**2)

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
            fraction_mc = EOE / mc_energy
            fraction_hits.append(fraction_mc)
            fraction_clus.append(cluster_energy[i][0] / mc_energy)
        fraction_hits = np.array(fraction_hits)
        fraction_clus = np.array(fraction_clus)
        median = np.median(fraction_hits)
        q16, q84 = np.percentile(fraction_hits, [16, 84])
        hits_low.append(median-q16)
        hits_high.append(q84-median)
        hits_median.append(median)
        median = np.median(fraction_clus)
        q16, q84 = np.percentile(fraction_clus, [16, 84])
        clus_low.append(median-q16)
        clus_high.append(q84-median)
        clus_median.append(median)
#Ok now we make a graph across the varing angles



#Summary graph, number of bib clusters given 1 normal cluster
plt.errorbar(angles, hits_median, yerr=[hits_low, hits_high], fmt='s', alpha= 0.6, capsize=4, label=f"Hit Energy Response")
plt.errorbar(angles, clus_median, yerr=[clus_low, clus_high], fmt='s', alpha= 0.6, capsize=4, label=f"Cluster Energy Response")
plt.xlabel("Varying Angles (Degrees)")
plt.ylabel("Response")
plt.title(f" Response comparing Cluster and Hit Energy {pid}")
#plt.title(f" Original 50 GeV Median Response 1 Nobib Cluster Pid: {pid} ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"summary_response_hitclus_11.pdf")
plt.close()