#This is checking the PFOS reconstruction energy

#We are now going to do fraction w/ all of the clusters
#Comparing energy to total energy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot
choices = [50]

for num in choices:
    #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_85-85_fedecheck/job_0/reco_output_p__nobib_oldalg.edm4hep.root")
    #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_11_pt_50_theta_25-25_testing_fede/job_0/reco_output_p50_11_nobib0.edm4hep.root")
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_25-25_test/job_0/reco_output_p50_211_nobib0.edm4hep.roo")
    '''
    meta_tree = file["podio_metadata"]
    meta = meta_tree["events___CollectionTypeInfo"]

    ids   = meta["events___CollectionTypeInfo.collectionID"].array()[0]
    names = meta["events___CollectionTypeInfo.name"].array()[0]
    types = meta["events___CollectionTypeInfo.dataType"].array()[0]

    collid2name = {int(i): str(n) for i, n in zip(ids, names)}

    for i, n, t in zip(ids, names, types):
        print(i, n, t)
    ''' 
    print("this is 25 degrees")
    events = file["events"]
    clusters = events["PandoraClusters"]
    mcparticles = events["MCParticles"]
    cluster_energies = clusters["PandoraClusters.energy"].array()
    pdg_momentum_x = mcparticles["MCParticles.momentum.x"].array()
    pdg_momentum_y = mcparticles["MCParticles.momentum.y"].array()
    pdg_momentum_z = mcparticles["MCParticles.momentum.z"].array()
    statuses = mcparticles["MCParticles.generatorStatus"].array()
    masses = mcparticles["MCParticles.mass"].array()
    pfo_energy = events["PandoraPFOs/PandoraPFOs.energy"].array()
    pandora_tracks_begin = events["PandoraPFOs/PandoraPFOs.tracks_begin"].array()
    pandora_tracks_end   = events["PandoraPFOs/PandoraPFOs.tracks_end"].array()
    pandora_clusters_begin = events["PandoraPFOs/PandoraPFOs.clusters_begin"].array()
    pandora_clusters_end   = events["PandoraPFOs/PandoraPFOs.clusters_end"].array()
    pandora_clusters_index = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.index"].array()
    pandora_clusters_collection = events["_PandoraPFOs_clusters/_PandoraPFOs_clusters.collectionID"].array()
    pandora_tracks_index = events["_PandoraPFOs_tracks/_PandoraPFOs_tracks.index"].array()
    pandora_tracks_collection = events["_PandoraPFOs_tracks/_PandoraPFOs_tracks.collectionID"].array()
    pandora_track_statesbegin = events["AllTracks/AllTracks.trackStates_begin"].array()
    pandora_track_statesend = events["AllTracks/AllTracks.trackStates_end"].array()
    cluster_energies = events["PandoraClusters/PandoraClusters.energy"].array()  # adjust name as needed
    track_states_phi = events["_AllTracks_trackStates/_AllTracks_trackStates.phi"].array()
    track_states_omega = events[ "_AllTracks_trackStates/_AllTracks_trackStates.omega" ].array()
    track_states_tanLambda = events[ "_AllTracks_trackStates/_AllTracks_trackStates.tanLambda" ].array()
    pfo_momx = events["PandoraPFOs/PandoraPFOs.momentum.x"].array()
    pfo_momy = events["PandoraPFOs/PandoraPFOs.momentum.y"].array()
    pfo_momz = events["PandoraPFOs/PandoraPFOs.momentum.z"].array()
    B = 5
    #for i in range(100):
    for i in range(events.num_entries):
        print(i)
        print(f"Number of clusters: {len(cluster_energies[i])}")
        print(f"Number of pfos: {len(pfo_energy[i])}")
        print(pandora_clusters_begin[i])
        print(pandora_clusters_end[i])
        print(pandora_clusters_index[i])
        print(pandora_clusters_collection[i])
        print(pandora_tracks_begin[i])
        print(pandora_tracks_end[i])
        print(pandora_tracks_index[i])
        print(pandora_tracks_collection[i])
        print(pandora_track_statesbegin[i])
        print(pandora_track_statesend[i])
        # Loop over PFOs
        num_matched_tracks = 0
        num_total_tracks = len(pandora_track_statesbegin[i])
        matched_pfos = 0 
        num_total_pfos = len(pfo_energy[i])
        for pfo in range(len(pfo_energy[i])):

            print(f"\n--- PFO {pfo} ---")

            # Track relations belonging to this PFO
            track_begin = pandora_tracks_begin[i][pfo]
            track_end = pandora_tracks_end[i][pfo]

            print(f"Track relations: {track_begin} -> {track_end}")
            has_track = False
            # Loop over tracks associated with this PFO
            
            for j in range(track_begin, track_end):
                
                track_index = pandora_tracks_index[i][j]
                collection_id = pandora_tracks_collection[i][j]

                print(f"  Track relation {j}")
                print(f"    collectionID = {collection_id}")
                print(f"    track index  = {track_index}")
                
                # Only look at AllTracks
                if collection_id == 4111979324:
                    num_matched_tracks += 1
                    has_track = True
                    state_begin = pandora_track_statesbegin[i][track_index]
                    state_end = pandora_track_statesend[i][track_index]
                    phi_track = track_states_phi[i][state_begin]
                    omega_track = track_states_omega[i][state_begin]
                    tanlambda_track = track_states_tanLambda[i][state_begin]
                    pT = abs(0.3 * B / omega_track / 1000)
                    theta = np.arctan2(1.0, tanlambda_track)
                    print(f" pT = {pT:.4f} GeV") 
                    print(f" theta = {theta:.4f} rad")
                    print(f" theta = {np.degrees(theta):.2f} deg") 
                    print(f" phi = {phi_track:.4f} rad") 
                    print(f" phi = {np.degrees(phi_track):.2f} deg")
            if has_track: 
                matched_pfos += 1
        #fraction_tracks = num_matched_tracks / num_total_tracks
        #fraction_pfos = matched_pfos / num_total_pfos
        #print(fraction_tracks)
        #print(fraction_pfos)
                    


