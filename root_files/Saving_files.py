#Getting clusters/points

import uproot
import numpy as np
import pickle
#Will probably have to download pickle too 

#Things to keep
#number tracks, xyz tracks, maybe the mc particles
#Ok I will have to choose 
#Events
electron_1=[0,50,269,2678]
electron_2=[0,100,2663,6595]
#electron_5=[0,32,9400,9664,9748]
#electron_10=[0,3071,4000,4536]
#electron_50=[0,1527,2211,3116,9948]
#electron_100=[0,436,500,1289,7401,9228]
#electron_150=[0,370,2978,5690,9900]
#electron_200=[980,2497,6919,9940]

electrons = {
    1: electron_1,
    2: electron_2,
    #5: electron_5,
    #10: electron_10,
    #50: electron_50,
    #100: electron_100,
    #150: electron_150,
    #200: electron_200
}
pion_1=[0,64,2525,3506,4392,9950]
pion_2=[1,286,2075,6504,9930]
#pion_5=[0,3611,9900]
#pion_10=[0,1417,7290,9900]
#lister = list(range(5900, 6001))
#pion_50=[0,1435,3422,3508,6015,6030,6874] #maybe 6015 normal
#pion_50.extend(lister) #Use extend not append
#pion_100=[1,446,8485]
#pion_150=[0,1,771,5531,5785,6288,9114]
#pion_200=[0,2,1124,1434,2029]
pions = {
    1: pion_1,
    2: pion_2,
    #5: pion_5,
    #10: pion_10,
    #50: pion_50,
    #100: pion_100,
    #150: pion_150,
    #200: pion_200
}

#Ok then i want for 50 5900-6000


#choices = [1,2,5,10,50,100,150,200]'
choices = [1,2]
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
all_data = {}
for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_{num}_theta_15-15/reco_pdg_11_pt_{num}_theta_15-15.root")
    list_to_look = electrons[num]
    events = file["events"]
    event_indices = list_to_look
    clusters = events["PandoraClusters"]
    cluster_data = {
        "cluster_x" : clusters["PandoraClusters.position.x"].array()[event_indices],
        "cluster_y" : clusters["PandoraClusters.position.y"].array()[event_indices],
        "cluster_z" : clusters["PandoraClusters.position.z"].array()[event_indices],
        "cluster_energy" : clusters["PandoraClusters.energy"].array()[event_indices],
    }
    pandora_clusters = events["_PandoraClusters_hits"]
    hit_index_all = pandora_clusters["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters["_PandoraClusters_hits.collectionID"].array()
    # event clusterf hit ranges
    hits_begin_all = clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = clusters["PandoraClusters.hits_end"].array()
    pandora_cluster_data = {}
    for evt in event_indices:
        pandora_cluster_data[evt] = []
        begins = hits_begin_all[evt]
        ends   = hits_end_all[evt]
        for b, e in zip(begins, ends):
            b = int(b)
            e = int(e)
            pandora_cluster_data[evt].append({
                "hit_index": hit_index_all[b:e],
                "collectionID": collectionID_all[b:e]
            })
    #all_tracks = events["AllTracks"]
    #deduped_tracks = events["DedupedTracks_objIdx"]
    #deduped_tracks_data = {
    #    "index_deduped" : deduped_tracks["DedupedTracks_objIdx.index"].array()[event_indices]
    #}
    mc_particles = events["MCParticles"]
    mc_particles_data = {
        "pdg_momentum_x" : mc_particles["MCParticles.momentum.x"].array()[event_indices],
        "pdg_momentum_y" : mc_particles["MCParticles.momentum.y"].array()[event_indices],
        "pdg_momentum_z" : mc_particles["MCParticles.momentum.z"].array()[event_indices],
        "statuses" : mc_particles["MCParticles.generatorStatus"].array()[event_indices],
        "masses" : mc_particles["MCParticles.mass"].array()[event_indices]
    }
    #track_states = events["_AllTracks_trackStates"]
    #track_states_data = {
    #    "tan_lambda" : track_states["_AllTracks_trackStates.tanLambda"].array()[event_indices]
    #}
    subsystem_hitmap = {}
    for name in real_systems:
        prefix = f"{name}/{name}"
        subsystem_hitmap[name] = {
            "posx": events[f"{prefix}.position.x"].array()[event_indices],
            "posy": events[f"{prefix}.position.y"].array()[event_indices],
            "posz": events[f"{prefix}.position.z"].array()[event_indices],
            "energy_map": events[f"{prefix}.energy"].array()[event_indices],
            "times": events[f"{prefix}.time"].array()[event_indices],
        }
    
    all_data[num] = {
        "clusters": cluster_data,
        #"all_tracks": all_tracks,
        "mc_particles": mc_particles_data,
        "subsystems": subsystem_hitmap,
        #"deduped_tracks": deduped_tracks_data,
        "pandora_clusters": pandora_cluster_data
    }
with open("/users/rldohert/data/mucoll/rldohert/electron_subset.pkl", "wb") as f:
    pickle.dump(all_data, f)


all_data={}
for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15/reco_pdg_211_pt_{num}_theta_15-15.root")
    list_to_look = pions[num]
    events = file["events"]
    event_indices = list_to_look

    #First all the clustering things
    clusters = events["PandoraClusters"]
    cluster_data = {
        "cluster_x" : clusters["PandoraClusters.position.x"].array()[event_indices],
        "cluster_y" : clusters["PandoraClusters.position.y"].array()[event_indices],
        "cluster_z" : clusters["PandoraClusters.position.z"].array()[event_indices],
        "cluster_energy" : clusters["PandoraClusters.energy"].array()[event_indices],
    }
    pandora_clusters = events["_PandoraClusters_hits"]
    hit_index_all = pandora_clusters["_PandoraClusters_hits.index"].array()
    collectionID_all = pandora_clusters["_PandoraClusters_hits.collectionID"].array()
    # event cluster hit ranges
    hits_begin_all = clusters["PandoraClusters.hits_begin"].array()
    hits_end_all   = clusters["PandoraClusters.hits_end"].array()
    pandora_cluster_data = {}
    for evt in event_indices:
        pandora_cluster_data[evt] = []
        begins = hits_begin_all[evt]
        ends   = hits_end_all[evt]
        for b, e in zip(begins, ends):
            b = int(b)
            e = int(e)
            pandora_cluster_data[evt].append({
                "hit_index": hit_index_all[b:e],
                "collectionID": collectionID_all[b:e]
            })
    #all_tracks = events["AllTracks"]
    #deduped_tracks = events["DedupedTracks_objIdx"]
    #deduped_tracks_data = {
    #    "index_deduped" : deduped_tracks["DedupedTracks_objIdx.index"].array()[event_indices]
    #}
    mc_particles = events["MCParticles"]
    mc_particles_data = {
        "pdg_momentum_x" : mc_particles["MCParticles.momentum.x"].array()[event_indices],
        "pdg_momentum_y" : mc_particles["MCParticles.momentum.y"].array()[event_indices],
        "pdg_momentum_z" : mc_particles["MCParticles.momentum.z"].array()[event_indices],
        "statuses" : mc_particles["MCParticles.generatorStatus"].array()[event_indices],
        "masses" : mc_particles["MCParticles.mass"].array()[event_indices]
    }
    #track_states = events["_AllTracks_trackStates"]
    #track_states_data = {
    #    "tan_lambda" : track_states["_AllTracks_trackStates.tanLambda"].array()[event_indices] 
    #}
    subsystem_hitmap = {} #define whay holds
    for name in real_systems: #going through the loops
        prefix = f"{name}/{name}" #getting prefix
        subsystem_hitmap[name] = { #Actually holding the data 
            "posx": events[f"{prefix}.position.x"].array()[event_indices],
            "posy": events[f"{prefix}.position.y"].array()[event_indices],
            "posz": events[f"{prefix}.position.z"].array()[event_indices],
            "energy_map": events[f"{prefix}.energy"].array()[event_indices],
            "times": events[f"{prefix}.time"].array()[event_indices],
        }

    all_data[num] = {
        "clusters": cluster_data,
        #"deduped_tracks": deduped_tracks_data,
        "mc_particles": mc_particles_data,
        "subsystems": subsystem_hitmap,
        #"all_tracks": all_tracks,
        "pandora_clusters": pandora_cluster_data
    }
with open("/users/rldohert/data/mucoll/rldohert/pions_subset.pkl", "wb") as f:
    pickle.dump(all_data, f)
#Warning all tracks 

'''
#!/bin/bash
#SBATCH -J distance
#SBATCH -p batch
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -o distance_%j.out
#SBATCH -e distance_%j.err

# Load the correct python module
module load python/3.11.0s-ixrhc3q
export PATH=$HOME/.local/bin:$PATH

cd /users/rldohert/work/mucoll/mucoll-slurm/files
# Run your python script
python -u Distance_big_root.py
'''