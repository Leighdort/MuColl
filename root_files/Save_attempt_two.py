#Getting clusters/points

import uproot
import numpy as np
import pickle
import pandas as pd
#Will probably have to download pickle too 
system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec"
}

electron_1=[0,50,269,2678]
electron_2=[0,100,2663,6595]


electrons = {
    1: electron_1,
    2: electron_2,
}
pion_1=[0,64,2525,3506,4392,9950]
pion_2=[1,286,2075,6504,9930]

pions = {
    1: pion_1,
    2: pion_2,
}

choices = [1,2]
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
all_data = {}
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_1_theta_15-15/reco_pdg_11_pt_1_theta_15-15.root")

#Let's just start with event1
events = file["events"]
mcparticles = events["MCParticles"]
deduped_tracks = events["DedupedTracks_objIdx"]

i=0
particle_ids = mcparticles["MCParticles.PDG"].array()[i] #This is total ids
pdg_momentum_x = mcparticles["MCParticles.momentum.x"].array()[i]
pdg_momentum_y = mcparticles["MCParticles.momentum.y"].array()[i]
pdg_momentum_z = mcparticles["MCParticles.momentum.z"].array()[i]
mass = mcparticles["MCParticles.mass"].array()[i]
total_momentum = (pdg_momentum_x**2 + pdg_momentum_y**2 + pdg_momentum_z**2)**0.5
total_energy_fast = np.sqrt(total_momentum**2 + mass**2) #This is a list of total energy
status_gen = mcparticles["MCParticles.generatorStatus"].array()[i]
num_tracks = len(deduped_tracks["DedupedTracks_objIdx.index"].array()[i])
clusters = events["PandoraClusters"]
pch = events["_PandoraClusters_hits"]
hit_index_all    = pch["_PandoraClusters_hits.index"].array()
collectionID_all = pch["_PandoraClusters_hits.collectionID"].array()
hits_begin_all = clusters["PandoraClusters.hits_begin"].array()
hits_end_all   = clusters["PandoraClusters.hits_end"].array()
# Choose event i
hits_begin_arr = hits_begin_all[i]
hits_end_arr   = hits_end_all[i]
hit_index      = hit_index_all[i]
collection_ID  = collectionID_all[i]

posx = {}
posy = {}
posz = {}
energy = {}
for name in system2name.values():
    prefix = f"{name}/{name}"
    posx[name]   = events[f"{prefix}.position.x"].array()
    posy[name]   = events[f"{prefix}.position.y"].array()
    posz[name]   = events[f"{prefix}.position.z"].array()
    energy[name] = events[f"{prefix}.energy"].array()

clusters_list = []

num_clusters = len(hits_begin_arr)
        
for j in range(num_clusters):
        
    hits_begin = hits_begin_arr[j]
    hits_end   = hits_end_arr[j]
        
    cluster_hit_indices    = hit_index[hits_begin:hits_end]
    cluster_collection_IDs = collection_ID[hits_begin:hits_end]
    xs, ys, zs, weights = [], [], [], []
    for idx, coll_id in zip(cluster_hit_indices, cluster_collection_IDs):
        if (coll_id == 3403901740):
            continue
        sysname = system2name[coll_id]
        xs.append(posx[sysname][i][idx])
        ys.append(posy[sysname][i][idx])
        zs.append(posz[sysname][i][idx])
        weights.append(energy[sysname][i][idx])
        # Store this cluster
    clusters_list.append({
        "x": np.array(xs),
        "y": np.array(ys),
        "z": np.array(zs),
        "energy": np.array(weights)
    })

all_x = []
all_y = []
all_z = []
cluster_id = []
energy_here = []
for j, c in enumerate(clusters_list):
    all_x.extend(c["x"])
    all_y.extend(c["y"])
    all_z.extend(c["z"])
    cluster_id.extend([j] * len(c["x"]))
    energy_here.extend(c["energy"])

df = pd.DataFrame({
    "x": all_x,
    "y": all_y,
    "z": all_z,
    "cluster": cluster_id,
    "energy": energy_here
})

all_data = {
    "mcparticle_ids" : particle_ids,
    "mcparticle_momentum": total_momentum,
    "mcparticle_energy": total_energy_fast,
    "mcparticle_status": status_gen,
    "num_tracks": num_tracks,
    "hit_dataframe": df,
    "event_num": i,
    "energy_event": 1 ,
    "event tag": "electron"
}

with open("electron_energy1_2.pkl", "wb") as f:
    pickle.dump(all_data, f)
