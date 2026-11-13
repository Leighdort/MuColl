
import numpy as np
import uproot

FILE = ("/users/rldohert/data/mucoll/rldohert/"
        "pdg_211_pt_50_theta_85-85_testing_fede2/job_0/"
        "reco_output_p50_211_nobib0.edm4hep.root")

EVENTS = list(range(940, 950))
N_PRINT = 3

file = uproot.open(FILE)
events = file["events"]

# ------------------------------------------------------------ cluster arrays
cluster_energy = events["PandoraClusters/PandoraClusters.energy"].array()
hits_begin     = events["PandoraClusters/PandoraClusters.hits_begin"].array()
hits_end       = events["PandoraClusters/PandoraClusters.hits_end"].array()

hit_index      = events["_PandoraClusters_hits/_PandoraClusters_hits.index"].array()
hit_collection = events["_PandoraClusters_hits/_PandoraClusters_hits.collectionID"].array()

# ------------------------------------------------------------ calorimeter arrays
# Collection IDs map cluster hits to their calorimeter collections.
systems = {
    679272617:  "EcalBarrel",
    1573202488: "HcalBarrel",
    3383333369: "EcalEndcap",
    2381985645: "HcalEndcap",
}

# ------------------------------------------------------------ ECal Barrel
eb_rec_energy = events["EcalBarrelCollectionRec/EcalBarrelCollectionRec.energy"].array()
eb_rec_cell   = events["EcalBarrelCollectionRec/EcalBarrelCollectionRec.cellID"].array()
eb_rel_from   = events["_EcalBarrelRelationsSimRec_from/_EcalBarrelRelationsSimRec_from.index"].array()
eb_rel_to     = events["_EcalBarrelRelationsSimRec_to/_EcalBarrelRelationsSimRec_to.index"].array()
eb_sim_energy = events["ECalBarrelCollection/ECalBarrelCollection.energy"].array()
eb_sim_cell   = events["ECalBarrelCollection/ECalBarrelCollection.cellID"].array()
eb_weight = events["EcalBarrelRelationsSimRec/EcalBarrelRelationsSimRec.weight"].array()
# ------------------------------------------------------------ HCal Barrel
hb_rec_energy = events["HcalBarrelCollectionRec/HcalBarrelCollectionRec.energy"].array()
hb_rec_cell   = events["HcalBarrelCollectionRec/HcalBarrelCollectionRec.cellID"].array()
hb_rel_from   = events["_HcalBarrelRelationsSimRec_from/_HcalBarrelRelationsSimRec_from.index"].array()
hb_rel_to     = events["_HcalBarrelRelationsSimRec_to/_HcalBarrelRelationsSimRec_to.index"].array()
hb_sim_energy = events["HCalBarrelCollection/HCalBarrelCollection.energy"].array()
hb_sim_cell   = events["HCalBarrelCollection/HCalBarrelCollection.cellID"].array()
hb_weight = events["HcalBarrelRelationsSimRec/HcalBarrelRelationsSimRec.weight"].array()

# ------------------------------------------------------------ ECal Endcap
ee_rec_energy = events["EcalEndcapCollectionRec/EcalEndcapCollectionRec.energy"].array()
ee_rec_cell   = events["EcalEndcapCollectionRec/EcalEndcapCollectionRec.cellID"].array()
ee_rel_from   = events["_EcalEndcapRelationsSimRec_from/_EcalEndcapRelationsSimRec_from.index"].array()
ee_rel_to     = events["_EcalEndcapRelationsSimRec_to/_EcalEndcapRelationsSimRec_to.index"].array()
ee_sim_energy = events["ECalEndcapCollection/ECalEndcapCollection.energy"].array()
ee_sim_cell   = events["ECalEndcapCollection/ECalEndcapCollection.cellID"].array()
ee_weight = events["EcalEndcapRelationsSimRec/EcalEndcapRelationsSimRec.weight"].array()
# ------------------------------------------------------------ HCal Endcap
he_rec_energy = events["HcalEndcapCollectionRec/HcalEndcapCollectionRec.energy"].array()
he_rec_cell   = events["HcalEndcapCollectionRec/HcalEndcapCollectionRec.cellID"].array()
he_rel_from   = events["_HcalEndcapRelationsSimRec_from/_HcalEndcapRelationsSimRec_from.index"].array()
he_rel_to     = events["_HcalEndcapRelationsSimRec_to/_HcalEndcapRelationsSimRec_to.index"].array()
he_sim_energy = events["HCalEndcapCollection/HCalEndcapCollection.energy"].array()
he_sim_cell   = events["HCalEndcapCollection/HCalEndcapCollection.cellID"].array()
he_weight = events["HcalEndcapRelationsSimRec/HcalEndcapRelationsSimRec.weight"].array()

# ------------------------------------------------------------ organize arrays
#Test factors
calibration_mip = 0.0001575
calibration_factorsMipGev = 0.00641222630095

factor_to_multiply = calibration_factorsMipGev / calibration_mip
data = {
    679272617: {
        "name": "EcalBarrel",
        "rec_energy": eb_rec_energy,
        "rec_cell": eb_rec_cell,
        "rel_from": eb_rel_from,
        "rel_to": eb_rel_to,
        "sim_energy": eb_sim_energy,
        "sim_cell": eb_sim_cell,
        "rel_weight": eb_weight
    },
    1573202488: {
        "name": "HcalBarrel",
        "rec_energy": hb_rec_energy,
        "rec_cell": hb_rec_cell,
        "rel_from": hb_rel_from,
        "rel_to": hb_rel_to,
        "sim_energy": hb_sim_energy,
        "sim_cell": hb_sim_cell,
        "rel_weight": hb_weight
    },
    3383333369: {
        "name": "EcalEndcap",
        "rec_energy": ee_rec_energy,
        "rec_cell": ee_rec_cell,
        "rel_from": ee_rel_from,
        "rel_to": ee_rel_to,
        "sim_energy": ee_sim_energy,
        "sim_cell": ee_sim_cell,
        "rel_weight": ee_weight
    },
    2381985645: {
        "name": "HcalEndcap",
        "rec_energy": he_rec_energy,
        "rec_cell": he_rec_cell,
        "rel_from": he_rel_from,
        "rel_to": he_rel_to,
        "sim_energy": he_sim_energy,
        "sim_cell": he_sim_cell,
        "rel_weight": he_weight
    },
}

# ------------------------------------------------------------ event loop
evt = 948

for f, t, w in zip(
    eb_rel_from[evt][:10],
    eb_rel_to[evt][:10],
    eb_weight[evt][:10]
):
    print(f"from={f}, to={t}, weight={w}")

for evt in EVENTS:

    print("EVENT", evt)
    printed = 0
    n_no_relation = 0
    n_cell_mismatch = 0
    lookup = {}

    for code, d in data.items():
        rec_to_sim = {}

        for f, t, w in zip(d["rel_from"][evt], d["rel_to"][evt], d["rel_weight"][evt]):
            rec_to_sim.setdefault(int(f), []).append((int(t), float(w)))

        lookup[code] = rec_to_sim

    #Now looping over clusters in this event 
    for c in range(len(cluster_energy[evt])):
        lo = int(hits_begin[evt][c])
        hi = int(hits_end[evt][c])

        E_rec_total = 0.0
        E_sim_total = 0.0

        print(
            f"\n  cluster {c}: E = {cluster_energy[evt][c]:.3f}, "
            f"hits {lo}:{hi}"
        )
        seen_sim = set()
        for h in range(lo, hi):
            

            code = int(hit_collection[evt][h]) #collection
            rec_index = int(hit_index[evt][h]) #Hit it referse to 

            if code not in data:
                #These are just hte muon hits
                continue

            d = data[code] #just looking at the hit collection

            # Reconstructed hit
            rec_E = d["rec_energy"][evt][rec_index] #Getting reco_E energy 
            rec_cell = d["rec_cell"][evt][rec_index] #Checking the cell_ID added check
            E_rec_total += rec_E

            # Reconstructed hit -> simulated hit(s)
            sim_list = lookup[code].get(rec_index, [])
            print(f" Cluster hit {h}: Rec hit {rec_index}, E = {rec_E:.5f}")

            if len(sim_list) == 0:
                n_no_relation += 1
                print(f"No Simulated hit relation")

            for sim_index, weight in sim_list:
                sim_E = d["sim_energy"][evt][sim_index]
                weighted_sim_E = sim_E * weight * factor_to_multiply
                E_sim_total+= weighted_sim_E
                    
                
                sim_cell = d["sim_cell"][evt][sim_index]

                if rec_cell != sim_cell:
                    n_cell_mismatch += 1

                if printed < N_PRINT:
                    ratio = rec_E / sim_E if sim_E != 0 else float("inf")

                    print(
                        f"     hit {h}: {d['name']} rec#{rec_index} "
                        f"E={rec_E:.5f} -> sim#{sim_index} E={weighted_sim_E:.6f} "
                        f"(rec/sim = {ratio:.1f}, "
                        f"cellID match = {rec_cell == sim_cell})"
                    )
                    printed += 1

        print(
            f"     SUM: E_rec_hits = {E_rec_total:.4f}, "
            f"E_sim = {E_sim_total:.5f}, "
            f"n_hits = {hi - lo}, "
            f"no relation = {n_no_relation}, "
            f"cellID mismatches = {n_cell_mismatch}, "
        )



import matplotlib.pyplot as plt
import numpy as np

# Proposed conversion from simulated energy to reconstructed-energy scale
calibration_mip = 0.0001575
calibration_factorsMipGev = 0.00641222630095
factor_to_multiply = 1 / (calibration_mip / calibration_factorsMipGev)

# Store one relative difference for each cluster
relative_differences = []
cluster_event_ids = []
cluster_ids = []

n_zero_sim_energy = 0
n_no_relation = 0

# Loop over every event in the file
n_events = len(cluster_energy)

for evt in range(n_events):

    # Map reconstructed-hit index -> corresponding simulated-hit index(es)
    lookup = {}

    for code, d in data.items():
        rec_to_sim = {}

        for f, t, w in zip(
            d["rel_from"][evt],
            d["rel_to"][evt],
            d["rel_weight"][evt]
        ):
            rec_to_sim.setdefault(int(f), []).append((int(t), float(w)))

        lookup[code] = rec_to_sim

    # Loop over Pandora clusters in this event
    for c in range(len(cluster_energy[evt])):
        lo = int(hits_begin[evt][c])
        hi = int(hits_end[evt][c])

        E_rec_total = 0.0
        E_sim_total = 0.0

        # Avoid counting the same simulated hit more than once
        # within a cluster.
        seen_sim = set()

        for h in range(lo, hi):
            code = int(hit_collection[evt][h])
            rec_index = int(hit_index[evt][h])

            # Skip hits not belonging to the four calorimeter systems
            if code not in data:
                continue

            d = data[code]

            # Reconstructed hit energy
            rec_E = float(d["rec_energy"][evt][rec_index])
            E_rec_total += rec_E

            # Find its corresponding simulated hit(s)
            sim_list = lookup[code].get(rec_index, [])

            if not sim_list:
                n_no_relation += 1
                continue

            for sim_index, weight in sim_list:
                # Prevent double-counting if multiple cluster hits
                # link to the same simulated hit.
                sim_key = (code, sim_index)

                if sim_key in seen_sim:
                    continue

                seen_sim.add(sim_key)

                sim_E = float(d["sim_energy"][evt][sim_index])
                E_sim_total += sim_E * weight * factor_to_multiply

        # Relative difference for this cluster
        if E_sim_total <= 0:
            n_zero_sim_energy += 1
            continue

        relative_difference = (
            (E_rec_total - E_sim_total) / E_sim_total
        )

        relative_differences.append(relative_difference)
        cluster_event_ids.append(evt)
        cluster_ids.append(c)

relative_differences = np.asarray(relative_differences)

print(f"Events processed: {n_events}")
print(f"Clusters plotted: {len(relative_differences)}")
print(f"Clusters skipped due to zero simulated energy: {n_zero_sim_energy}")
print(f"Reconstructed hits without a SimRec relation: {n_no_relation}")
print(f"Mean relative difference: {np.mean(relative_differences):.4f}")
print(f"Median relative difference: {np.median(relative_differences):.4f}")

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 6))
bins = np.linspace(-0.15, 0.15, 60)
plt.hist(relative_differences, bins=bins, alpha=0.5,label='diff', edgecolor='black')
plt.xlabel("Response")
plt.ylabel("Count")
plt.xlim(-1, 1)
plt.title(f"Response")
plt.legend()
fig.savefig("cluster_relative_energy_difference.png", dpi=200)
plt.close(fig)

print("Saved plot: cluster_relative_energy_difference.png")