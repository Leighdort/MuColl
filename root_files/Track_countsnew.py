#Tracks

#Here we're trying to find track pT


#Number of Tracks and Corresponding Clusters

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
import math 

#Notes -> 
B_z = 5 #B/c Maia 
#We need state.omega and state.tanLambda
choices = [2, 10, 50]
#First I will see the ratio of particle w/ status 0, its energy to leading cluster energy
pt_median = []
pt_low = []
pt_high = []
bpt_median = []
bpt_low = []
bpt_high = []

#Lets just do a basic Track_countsnew.py

#Ok I want state.omega, state.tanLambda and B_z?
#Take the 0th track state 
for num in choices:
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    events = file["events"]
    clusters = events["PandoraClusters"]
    all_tracks = events["AllTracks"]
    deduped_tracks = events["DedupedTracks_objIdx"]
    index_deduped = deduped_tracks["DedupedTracks_objIdx.index"].array()
    track_states = events["_AllTracks_trackStates"]
    omega = track_states["_AllTracks_trackStates.omega"].array()
    tanLambda = track_states["_AllTracks_trackStates.tanLambda"].array()
    cluster_energy = clusters["PandoraClusters.energy"].array()
    print(f"Processing {num} GeV")
    number_tracks = 0
    pt_list = []
    eta_list = []
    for i in range(events.num_entries):
        t_omega = omega[i]
        t_tanLambda = tanLambda[i]
        indices = index_deduped[i]
        print (indices)
        print (indices(0))
        '''
        print()
        if len (indices) == 0:
            continue
        if len(indices) !=1:
            continue
        meaningful_track = indices[0] #The zeroth index track state? Is the track at the edge???
        number_tracks += number_tracks
        tomega = t_omega[meaningful_track]
        tanLambda = t_tanLambda[meaningful_track]
        #Make sure you can write this awkward?
        pT = abs(0.3 * B_z / tomega / 1000)
        eta = -math.log(math.tan(math.pi / 2 - math.atan(state.tanLambda)) / 2)
        pt_list.append(pT)
        eta_list.append(eta)
        #These are still lists, just the most meaningful lists
        #Track states????
        #Do I care where the track is located 
    pt = np.array(pt)
    q16, q84 = np.percentile(pt, [16, 84])
    median = np.median(pt)
    mc_low.append(pt-q16)
    mc_high.append(q84-pt)
    mc_median.append(pt)
    bins=30
    plt.hist(fraction, bins=bins, edgecolor = 'black')
    plt.xlabel(f"Pt")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"PT for Tracks with Pions {num}")
    plt.tight_layout()
    plt.savefig(f"pt_tracks_pions{num}GeV.pdf")
    plt.close()
    '''
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    for i in range(events.num_entries):
        indexes = index_deduped[i]
        cluster_energies = cluster_energy[i]
        num_clusters = len(cluster_energies)
        num_tracks = len(indexes)
        number_tracks.append(num_tracks)
        if (num_tracks !=1):
            print("Bib")
            print(i)
            print(f"tracks: {num_tracks}")
            print(f"clusters: {num_clusters}")
    num_tracks = np.array(number_tracks)
    bib_mean.append(np.median(num_tracks))
    median = np.median(num_tracks)
    q16, q84 = np.percentile(num_tracks, [16, 84])
    bib_low.append(median - q16)
    bib_high.append(q84 - median)
    bins = np.arange(np.min(num_tracks), np.max(num_tracks) + 2) - 0.5
    plt.hist(num_tracks, bins=bins, edgecolor='black')
    #Adding log scale
    plt.yscale("log")
    plt.xlabel("Number of Tracks per Event ")
    plt.ylabel("Count")
    plt.axvline(
        median,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"Median = {median:.2f}"
    )
    plt.legend()
    plt.title(f"Track Count {num} GeV Pions with Bib")
    plt.tight_layout()
    plt.savefig(f"num_tracks_pions_bib{num}GeV.pdf")
    plt.close()

