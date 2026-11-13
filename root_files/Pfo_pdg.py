#Checking PFO type for the weird event
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
choices = [50]
from matplotlib.colors import LogNorm
file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_testing_fede2/job_0/reco_output_p50_211_nobib0.edm4hep.root")
#Should check again if it has no clusters
#Also shoud compare it to nearby PFOS

events = file["events"]
clusters = events["PandoraClusters"]
pfo_energy = events["PandoraPFOs/PandoraPFOs.energy"].array()
pandora_clusters_begin = events["PandoraPFOs/PandoraPFOs.clusters_begin"].array()
pandora_clusters_end = events["PandoraPFOs/PandoraPFOs.clusters_end"].array()
pandora_clusters_index = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.index"].array()
pandora_clusters_collection = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.collectionID"].array()
pandora_pfo_ID = events["PandoraPFOs/PandoraPFOs.PDG"].array()
pandora_tracks_begin = events["PandoraPFOs/PandoraPFOs.tracks_begin"].array()
pandora_tracks_end   = events["PandoraPFOs/PandoraPFOs.tracks_end"].array()

list_events = [940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950]

#Also want to check tracks 
for i, index in enumerate(list_events):
    pfo_energies = pfo_energy[index]
    
    for pfo in range(len(pfo_energies)):
        print(f"Event: {index}")
        print(f"PFO: {pfo}")
        cluster_begin = pandora_clusters_begin[index][pfo]
        cluster_end = pandora_clusters_end[index][pfo]
        tracks_begin = pandora_tracks_begin[index][pfo]
        tracks_end = pandora_tracks_end[index][pfo]
        number_of_clusters = cluster_end - cluster_begin
        number_of_tracks = tracks_end - tracks_begin
        pfo_pdg = pandora_pfo_ID[index][pfo]
        print(f"Number of tracks: {number_of_tracks}")
        print(f"Number of clusters: {number_of_clusters}")
        print(f"Particle PDG: {pfo_pdg}")
