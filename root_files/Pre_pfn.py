
#Trying to do bib no bib tracking 
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

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec"
}


real_systems = [
    "EcalBarrelCollectionRec",
    "HcalBarrelCollectionRec",
    "EcalEndcapCollectionRec",
    "HcalEndcapCollectionRec"
]

#Making some Things that I can Later use
def make_voxels(x, y, z, step):
    return set(
        (int(xi / step), int(yi / step), int(zi / step))
        for xi, yi, zi in zip(x, y, z)
    )
def voxel_overlap(a, b):
    if len(b) == 0:
        return 0.0
    return len(a & b) / len(b)
clus_high = []
clus_low = []
clus_median = []
hits_high = []
hits_low = []
hits_median = []
particle = [211]
angles = [85]
energy_list = [50]
#angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for e in energy_list: N
    for pid in particle:
        for a in angles:
            print(f"\n=== Energy {a} Degrees ===")

            file = uproot.open(
                f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_clustertest2/reco_pdg_{pid}_pt_{e}_theta_{a}-{a}_nobib.root"
            )

            events = file["events"]
            pandora_clusters = events["PandoraClusters"]N
            pandora_clusters_hits = events["_PandoraClusters_hits"]
            pch = events["_PandoraClusters_hits"]
            clusters = events["PandoraClusters"]
            hit_index_all    = pch["_PandoraClusters_hits.index"].array()
            collectionID_all = pch["_PandoraClusters_hits.collectionID"].array()
            hits_begin_all   = clusters["PandoraClusters.hits_begin"].array()
            hits_end_all     = clusters["PandoraClusters.hits_end"].array()

            file_bib = uproot.open(
                    f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_clustertest2/reco_pdg_{pid}_pt_{e}_theta_{a}-{a}_bib.root"
                )
            
            events_bib = file_bib["events"]
            print(len(events))
            print(len(events_bib))
            pandora_clusters_bib = events_bib["PandoraClusters"]
            pandora_clusters_hits_bib = events_bib["_PandoraClusters_hits"]
            pch_bib = events_bib["_PandoraClusters_hits"]
            clusters_bib = events_bib["PandoraClusters"]
            hit_index_all_bib    = pch_bib["_PandoraClusters_hits.index"].array()
            collectionID_bib = pch_bib["_PandoraClusters_hits.collectionID"].array()
            hits_begin_all_bib   = clusters_bib["PandoraClusters.hits_begin"].array()
            hits_end_all_bib     = clusters_bib["PandoraClusters.hits_end"].array()
           
            for i in range(events.num_entries):
            #for i in range(1, 6):
                hits_begin_arr = hits_begin_all[i]
                hits_end_arr   = hits_end_all[i]
                hit_index      = hit_index_all[i]
                collection_ID  = collectionID_all[i]
                hits_begin_arr_bib = hits_begin_all_bib[i]
                hits_end_arr_bib = hits_end_all_bib[i]
                hit_index_bib = hit_index_all_bib[i]
                collection_ID_bib = collectionID_bib[i]

                # Load arrays once
                posx, posy, posz, energy_hits = {}, {}, {}, {}
                for name in system2name.values():
                    prefix = f"{name}/{name}"
                    posx[name] = events[f"{prefix}.position.x"].array()
                    posy[name] = events[f"{prefix}.position.y"].array()
                    posz[name] = events[f"{prefix}.position.z"].array()
                    energy_hits[name] = events[f"{prefix}.energy"].array()
                cluster_list_nobib = []
                for j in range(len(hits_begin_arr)):
                    xs, ys, zs, ws = [], [], [], []

                    for idx, coll_id in zip(
                        hit_index[hits_begin_arr[j]:hits_end_arr[j]],
                        collection_ID[hits_begin_arr[j]:hits_end_arr[j]]
                    ):
                        if coll_id not in system2name:
                            continue

                        sys = system2name[coll_id]
                        xs.append(posx[sys][i][idx])
                        ys.append(posy[sys][i][idx])
                        zs.append(posz[sys][i][idx])
                        ws.append(energy_hits[sys][i][idx])

                    if xs:
                        cluster_list_nobib.append({
                            "clus_num": j,
                            "x": np.array(xs),
                            "y": np.array(ys),
                            "z": np.array(zs),
                            "energy": np.array(ws),
                            "voxels": make_voxels(xs, ys, zs, step = 0.01)
                        })
                #Now in cluster_list each cluster should have stored their hits and energy for funsies 
                #Now bib clusters. 
                #Once we get the hits arrays, should sort the clusters in descending highest energ
                #Then check from highest energy downwards

                #Should probbaly find a way to close the previous file first?

                #print("here now")
                
                # Load arrays once
                posx, posy, posz, energy_hits = {}, {}, {}, {}
                for name in system2name.values():
                    prefix = f"{name}/{name}"
                    posx[name] = events_bib[f"{prefix}.position.x"].array()
                    posy[name] = events_bib[f"{prefix}.position.y"].array()
                    posz[name] = events_bib[f"{prefix}.position.z"].array()
                    energy_hits[name] = events_bib[f"{prefix}.energy"].array()
                cluster_list_bib = []
                for j in range(len(hits_begin_arr_bib)):
                    xs, ys, zs, ws = [], [], [], []

                    for idx, coll_id in zip(
                        hit_index_bib[hits_begin_arr_bib[j]:hits_end_arr_bib[j]],
                        collection_ID_bib[hits_begin_arr_bib[j]:hits_end_arr_bib[j]]
                    ):
                        if coll_id not in system2name:
                            continue

                        sys = system2name[coll_id]
                        xs.append(posx[sys][i][idx])
                        ys.append(posy[sys][i][idx])
                        zs.append(posz[sys][i][idx])
                        ws.append(energy_hits[sys][i][idx])

                    if xs:
                        cluster_list_bib.append({
                            "clus_num": j,
                            "x": np.array(xs),
                            "y": np.array(ys),
                            "z": np.array(zs),
                            "energy": np.array(ws),
                            "total_energy": np.sum(ws),
                            "voxels": make_voxels(xs, ys, zs, step = 0.01)
                        })
                cluster_list_bib.sort(key=lambda c: c["total_energy"], reverse=True)
                #(bias towards higher clusters)
                #-717.846985 positions specificity 
                matches = []
                threshold = 0.9
                saved = 0
                for m, nobib in enumerate(cluster_list_nobib):
                    matched = False
                    for j, bib in enumerate(cluster_list_bib):
                        overlap = voxel_overlap(bib["voxels"], nobib["voxels"])
                        if overlap >= threshold:
                            if overlap == 1:
                                if (len(bib["voxels"]) >= 4*len(nobib["voxels"])):
                                    saved += 1
                            else:
                                matches.append((m, cluster_list_bib[j]["clus_num"], overlap))
                                matched = True
                                break
                    if not matched:
                        matches.append((m, None, 0.0))
                #Ok this loops through all the events
                #Matches is defined per event
            #Now we have succsesfully loaded both arrays
            #Now we want to sort the bib clusters in order of decreasing energy 
                with open("matches_500_clustertest2.txt", "a") as f:
                    f.write(f"Event: {i}")
                    for m in matches:
                        f.write(f"{m}\n")
                #print("written")





    