#Location_hits.py
'''
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

# System IDs (use directly, no dict lookup in loop)
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

e = 50
particle = [211, 11]
angles = [15, 85, 140]
#angles = [15, 85, 143]

for pid in particle:
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")

        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial6/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_10_theta_{a}-{a}_bib2/reco_pdg_{pid}_pt_10_theta_{a}-{a}_nobib.root")
        events = file["events"]

        pandora_clusters = events["PandoraClusters"]
        pandora_clusters_hits = events["_PandoraClusters_hits"]
        mcparticles = events["MCParticles"]

        # Load arrays once
        status_mc = mcparticles["MCParticles.generatorStatus"].array()

        cluster_energy = pandora_clusters["PandoraClusters.energy"].array()
        cluster_x = pandora_clusters["PandoraClusters.position.x"].array()
        cluster_y = pandora_clusters["PandoraClusters.position.y"].array()
        cluster_z = pandora_clusters["PandoraClusters.position.z"].array()

        hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
        hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

        collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
        hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()

        # Preload energy maps (unchanged)
        energy_map = {}
        for name in real_systems:
            prefix = f"{name}/{name}"
            energy_map[name] = events[f"{prefix}.energy"].array()

        ecal_hits = []
        hcal_hits = []
        ecal_ratio = []
        events_all = 0

        for i in range(events.num_entries):

            # Filter: exactly 1 cluster
            if len(cluster_energy[i]) != 1:
                continue

            # Pull per-event arrays ONCE
            hits_begin_arr = hits_begin_all[i]
            hits_end_arr   = hits_end_all[i]
            collection_ID  = collectionID_all[i]

            # Accumulators
            ecal_barrel = 0
            hcal_barrel = 0
            ecal_endcap = 0
            hcal_endcap = 0

            # Loop over clusters (usually 1 here anyway)
            for j in range(len(hits_begin_arr)):
                lo = hits_begin_arr[j]
                hi = hits_end_arr[j]

                sysIDs = collection_ID[lo:hi]

                # 🔥 VECTORIZED COUNTING (major speedup)
                ecal_barrel += np.sum(sysIDs == ECAL_BARREL)
                hcal_barrel += np.sum(sysIDs == HCAL_BARREL)
                ecal_endcap += np.sum(sysIDs == ECAL_ENDCAP)
                hcal_endcap += np.sum(sysIDs == HCAL_ENDCAP)
            
            total = ecal_barrel + hcal_barrel + ecal_endcap + hcal_endcap
            ecal_total = ecal_barrel + ecal_endcap
            hcal_total = hcal_barrel + hcal_endcap
            print(total)
            print(hcal_total)

            ecal_hits.append(ecal_total)
            hcal_hits.append(hcal_total)

            if hcal_total != 0:
                events_all += 1

            if total != 0:
                ecal_ratio.append(ecal_total / total)
            else:
                ecal_ratio.append(0)

        print(f"For {e}, {a}, {pid}, we have {events_all} events in the hcal out of {events.num_entries}")

        # Convert to arrays (FIXED bug)
        ecal_hits = np.array(ecal_hits)
        hcal_hits = np.array(hcal_hits)
        total_hits = ecal_hits + hcal_hits #does this work as an array?
        hcal_ratio = hcal_hits / total_hits

        #Let's do a bar display of hcal_ratio

          # Define bins
        #Bin everything from 0 to 0.002
        #bins = 30
        bins = np.linspace(0, 0.002, 30)
        plt.xlim(0, .002)
        median = np.median(hcal_ratio)
        plt.hist(hcal_ratio, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Ratio hits in Hcal")
        plt.ylabel("Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.6f}"
        )
        plt.title(f"Hcal Ratio Hits {a} for PID: {pid} Energy 50 GeV")
        plt.tight_layout()
        plt.savefig(f"hcal_ratio_{a}_{pid}50Gev_2.pdf")
        plt.close()
'''
#Now we're going to do the proportion of energy in the hcal: 
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
#2 particles 11 211
#angles 143, 140, 15, 85 
#energy 2, 10, 50
particle = [211, 11]
#angles = [15, 85, 140]
angles = [15, 85, 143]
pids = [11, 211]
energies = [2, 10, 50]
anglees = [15, 85, 140, 143]

results = {
    pid: {
        E: {a: None for a in anglees}
        for E in energies
    }
    for pid in pids
}
for pid in particle:
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")

        #file = uproot.open(   f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial6/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_10_theta_{a}-{a}_bib2/reco_pdg_{pid}_pt_10_theta_{a}-{a}_nobib.root")
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

        fraction = []
        energy_ratio = []
        print((events.num_entries))
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
            energy_hcal = 0.0
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
                    energy_hcal += np.sum(energy_map["HcalBarrelCollectionRec"][i][idxs[mask_hb]])
                if np.any(mask_ee):
                    EOE += np.sum(energy_map["EcalEndcapCollectionRec"][i][idxs[mask_ee]])
                if np.any(mask_he):
                    EOE += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])
                    energy_hcal += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])
            energy_ratio.append(energy_hcal / EOE)
            fraction.append(EOE / mc_energy)
        fraction = np.array(fraction)
        energy_ratio = np.array(energy_ratio)
        '''
        # 📊 Histogram with log-scale y-axis
        plt.hist(fraction, bins=50, edgecolor='black')
        plt.yscale('log')   # ✅ log scale

        plt.xlabel("EOE / MC Energy")
        plt.ylabel("Count (log scale)")
        plt.title(f"{pid} at {a} degrees")

        plt.tight_layout()
        plt.savefig(f"fraction_{pid}_{a}.pdf")
        plt.close()
        '''
        #Mask for all energy_ratios < 0.002 (only want < 0.002)
        #count how many number
        #count how many zero now
        #Bin everything from 0 to 0.002
        # --- Mask: only values < 0.002 ---
        mask = energy_ratio < 0.003
        filtered = energy_ratio[mask]
        # --- Counts ---
        num_total = len(energy_ratio)
        num_below = np.sum(mask)
        ratiooooooo = num_below/num_total
        results[pid][10][a] = ratiooooooo
        print(f" For 10 GeV, pid {pid}, angle {a}")
        print(f"Total events: {num_total}")
        print(f"Events with energy_ratio < 0.003: {num_below}")
        results[pid][10][a] = ratiooooooo

        bins = np.linspace(0, 0.02, 60)
        plt.xlim(0, .02)
        median = np.median(energy_ratio)
        plt.hist(energy_ratio, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Ratio of energy hits in Hcal")
        plt.ylabel("Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.6f}"
        )
        plt.title(f"Hcal Ratio Energy Hits {a} for PID: {pid} Energy 10 GeV")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"hcal_energy_ratio_{a}_{pid}10Gev_fine.pdf")
        plt.close()

#Now for 2 GeV
angles = [15, 85, 143]
for pid in particle:
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")

        #file = uproot.open(   f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial6/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_2_theta_{a}-{a}_bib2/reco_pdg_{pid}_pt_2_theta_{a}-{a}_nobib.root")
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

        fraction = []
        energy_ratio = []
        print((events.num_entries))
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
            energy_hcal = 0.0
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
                    energy_hcal += np.sum(energy_map["HcalBarrelCollectionRec"][i][idxs[mask_hb]])
                if np.any(mask_ee):
                    EOE += np.sum(energy_map["EcalEndcapCollectionRec"][i][idxs[mask_ee]])
                if np.any(mask_he):
                    EOE += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])
                    energy_hcal += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])
            energy_ratio.append(energy_hcal / EOE)
            fraction.append(EOE / mc_energy)
        fraction = np.array(fraction)
        energy_ratio = np.array(energy_ratio)
        '''
        # 📊 Histogram with log-scale y-axis
        plt.hist(fraction, bins=50, edgecolor='black')
        plt.yscale('log')   # ✅ log scale

        plt.xlabel("EOE / MC Energy")
        plt.ylabel("Count (log scale)")
        plt.title(f"{pid} at {a} degrees")

        plt.tight_layout()
        plt.savefig(f"fraction_{pid}_{a}.pdf")
        plt.close()
        '''
        mask = energy_ratio < 0.003
        filtered = energy_ratio[mask]
        # --- Counts ---
        num_total = len(energy_ratio)
        num_below = np.sum(mask)
        print(f" For 2 GeV, pid {pid}, angle {a}")
        print(f"Total events: {num_total}")
        print(f"Events with energy_ratio < 0.003: {num_below}")
        ratiooooooo = num_below/num_total
        results[pid][2][a] = ratiooooooo
        
        #Bin everything from 0 to 0.002
        bins = np.linspace(0, 0.02, 60)
        plt.xlim(0, .02)
        median = np.median(energy_ratio)
        plt.hist(energy_ratio, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Ratio of energy hits in Hcal")
        plt.ylabel("Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.6f}"
        )
        plt.title(f"Hcal Ratio Energy Hits {a} for PID: {pid} Energy 2 GeV")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"hcal_energy_ratio_{a}_{pid}2Gev_fine.pdf")
        plt.close()

#Now for 50 GeV
angles = [15, 85, 140]
for pid in particle:
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")

        file = uproot.open(   f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial6/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_10_theta_{a}-{a}_bib2/reco_pdg_{pid}_pt_10_theta_{a}-{a}_nobib.root")
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

        fraction = []
        energy_ratio = []
        print((events.num_entries))
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
            energy_hcal = 0.0
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
                    energy_hcal += np.sum(energy_map["HcalBarrelCollectionRec"][i][idxs[mask_hb]])
                if np.any(mask_ee):
                    EOE += np.sum(energy_map["EcalEndcapCollectionRec"][i][idxs[mask_ee]])
                if np.any(mask_he):
                    EOE += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])
                    energy_hcal += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])
            energy_ratio.append(energy_hcal / EOE)
            fraction.append(EOE / mc_energy)
        fraction = np.array(fraction)
        energy_ratio = np.array(energy_ratio)
        '''
        # 📊 Histogram with log-scale y-axis
        plt.hist(fraction, bins=50, edgecolor='black')
        plt.yscale('log')   # ✅ log scale

        plt.xlabel("EOE / MC Energy")
        plt.ylabel("Count (log scale)")
        plt.title(f"{pid} at {a} degrees")

        plt.tight_layout()
        plt.savefig(f"fraction_{pid}_{a}.pdf")
        plt.close()
        '''
        mask = energy_ratio < 0.003
        filtered = energy_ratio[mask]
        # --- Counts ---
        num_total = len(energy_ratio)
        num_below = np.sum(mask)
        ratiooooooo = num_below/num_total
        results[pid][50][a] = ratiooooooo
        print(f" For 50 GeV, pid {pid}, angle {a}")
        print(f"Total events: {num_total}")
        print(f"Events with energy_ratio < 0.003: {num_below}")
        #Bin everything from 0 to 0.002
        bins = np.linspace(0, 0.02, 60)
        plt.xlim(0, .02)
        median = np.median(energy_ratio)
        plt.hist(energy_ratio, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Ratio of energy hits in Hcal")
        plt.ylabel("Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.6f}"
        )
        plt.title(f"Hcal Ratio Energy Hits {a} for PID: {pid} Energy 50 GeV")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"hcal_energy_ratio_{a}_{pid}50Gev_fine.pdf")
        plt.close()

for pid in [11, 211]:

    plt.figure()

    for a in anglees:
        y = [results[pid][E][a] for E in energies]

        plt.plot(
            energies,
            y,
            marker='o',
            label=f"{a}°"
        )

    plt.xlabel("Beam Energy (GeV)")
    plt.ylabel("Fraction HCAL ratio < 0.003")
    plt.title(f"PID {pid}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"pid_{pid}_hcal_vs_energy_by_angle.png")
    plt.close()

'''
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
#from scipy.optimize import curve_fit

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
energies = [50]
angles = [140, 141, 142, 143, 144, 145, 146]
#We would also have to loop through energies here but just wait first 
bounds = {
    211: {  # pion
        15: {2: 0.05, 10: 0.02, 30: 0.01, 50: 0.006},
        85: {2: 0.82, 10: 0.06, 30: 0.02, 50: 0.02},
        143:{2: 0.41, 10: 0.03, 30: 0.01, 50: 0.008},
    },
    11: {   # electron
        15: {2: 0.01, 10: 0.01, 30: 0.01, 50: 0.01},
        85: {2: 1.40, 10: 0.09, 30: 0.06, 50: 0.06},
        143:{2: 0.59, 5: 0.05, 10: 0.04, 15: 0.04, 30: 0.04, 50: 0.04},
    }
}
e = 50
particle = [11, 211]
angles = [15, 85, 140]
yes = True
for pid in particle:
    mc_median= []
    mc_low = []
    mc_high = []
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")
        fraction = []
        energy_clus = []
        energy_hits = []
        num_events_1_clus = 0
        passes = 0
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial6/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        events = file["events"]
        pandora_clusters = events["PandoraClusters"]
        #I need the theta phi of mc and of clusters s
        mcparticles = events["MCParticles"]
        #we want a status of 1 
        status_mc = mcparticles["MCParticles.generatorStatus"].array()
        #do I want vertex or endpoint? I presume endpoint and we will see both
        #I will try it first with endpoint
        pandora_clusters_hits = events["_PandoraClusters_hits"]
        collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
        mc_x = mcparticles["MCParticles.endpoint.x"].array()
        mc_y = mcparticles["MCParticles.endpoint.y"].array()
        mc_z = mcparticles["MCParticles.endpoint.z"].array()
        mc_momx = mcparticles["MCParticles.momentum.x"].array()
        mc_momy = mcparticles["MCParticles.momentum.y"].array()
        mc_momz = mcparticles["MCParticles.momentum.z"].array()
        mc_mass = mcparticles["MCParticles.mass"].array()
        #Now I want cluster energy
        #Also I want cluster theta, phi
        cluster_x=pandora_clusters["PandoraClusters.position.x"].array()
        cluster_y=pandora_clusters["PandoraClusters.position.y"].array()
        cluster_z=pandora_clusters["PandoraClusters.position.z"].array()
        cluster_energy=pandora_clusters["PandoraClusters.energy"].array()
        angular_dist = []
        regular_dist = []
        #Let's right now just filter
        #for i in range(events.num_entries):
        #for i in range((5001)):
        hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
        hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
        hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
        ecal_e_ratio = []
        hcal_e_ratio = []
        ecal_b_ratio = []
        hcal_b_ratio = []
        ecal_ratio = []
        ecal_hits = []
        hcal_hits = []
        events_all = 0
        energy_map = {}
        for name in real_systems:
            prefix = f"{name}/{name}"
            energy_map[name] = events[f"{prefix}.energy"].array()

        for i in range((events.num_entries)):
            #Let's just right now filter for events with only 1 cluster
            ecal_endcap = 0
            ecal_barrel = 0
            hcal_endcap = 0
            hcal_barrel = 0
            ecal_total = 0
            total = 0
            if len(cluster_energy[i]) == 1:
                num_events_1_clus +=1 
                clus_clus_energy = cluster_energy[i]
                mask = (status_mc[i] == 1)
                mx=mc_x[i][mask]
                my=mc_y[i][mask]
                mz=mc_z[i][mask]
                momx=mc_momx[i][mask]
                momy=mc_momy[i][mask]
                momz=mc_momz[i][mask]
                mcmass=mc_mass[i][mask]
                mc_r = np.sqrt(mx**2 + my**2 + mz**2)
                mc_theta = np.arccos(mz / mc_r) #these may all be in radians
                mc_phi = np.arctan2(my, mx)
                cx = cluster_x[i][0] #mind you, only works for 1 cluster
                cy = cluster_y[i][0]
                cz = cluster_z[i][0]
                c_r = np.sqrt(cx**2 + cy**2 + cz**2)
                c_theta = np.arccos(cz / c_r)
                c_phi = np.arctan2(cy, cx)
                cosang = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                cosang = np.clip(cosang, -1.0, 1.0)
                angular_distance = np.arccos(cosang)
                other_angle = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                other_dist = np.sqrt(mc_r**2 + c_r**2 - 2*mc_r*c_r*other_angle)
                #if angular_distance <= bound:
                if yes == True: 
                    passes += 1
                    mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                    mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                    #To find fraction we're going to have to recalibrate everything by hand "yay :)"
                    hits_begin_arr = hits_begin_all[i]
                    hits_end_arr = hits_end_all[i]
                    hit_index = hit_index_all[i]
                    collection_ID = collectionID_all[i]
                    EOE = 0.0 #energy of event
                    for j in range(len(hits_begin_arr)):
                        lo = hits_begin_arr[j]
                        hi = hits_end_arr[j]
                        sysIDs = collection_ID[lo:hi]
                        idxs = hit_index[lo:hi]
                        for p, code in enumerate(sysIDs):
                            name = system2name.get(code, "Skip")
                            if name == "Skip":
                                continue
                            #energy = energy_map[name][i][j]
                            #LET"S TEST BUT PROBBALY FAIL
                            idx = idxs[p]
                            energy = energy_map[name][i][idx]
                            if code == 679272617:
                                ecal_barrel +=1
                            elif code == 1573202488:
                                hcal_barrel +=1
                            elif code == 3383333369:
                                ecal_endcap +=1
                            elif code == 2381985645:
                                hcal_endcap +=1
                    total = ecal_barrel + hcal_barrel + ecal_endcap + hcal_endcap
                    ecal_total = ecal_barrel + ecal_endcap
                    hcal_total = hcal_barrel + hcal_endcap
                    ecal_hits.append(ecal_total)
                    hcal_hits.append(hcal_total)
                    if hcal_total != 0:
                        events_all+=1
                    if total != 0:
                        ecal_ratio.append(ecal_total / total)
                    else:
                        ecal_ratio.append(0)
        print(f"For {e}, {a}, {pid}, we have {events_all} events in the hcal out of {events.num_entries}")
        fraction = np.array(fraction)
        ecal_hits = np.array(ecal_hits)
        hcal_hits = np.array(hcal_total)
'''