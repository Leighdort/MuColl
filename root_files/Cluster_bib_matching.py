#Cluster Matching

#This is printing out specifics and also plotting resolution

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

system2name = {
    679272617: "EcalBarrelCollectionRec",
    1573202488: "HcalBarrelCollectionRec",
    3383333369: "EcalEndcapCollectionRec",
    2381985645: "HcalEndcapCollectionRec",
    3403901740: "Skip",
}
real_systems = ["EcalBarrelCollectionRec", "HcalBarrelCollectionRec","EcalEndcapCollectionRec", "HcalEndcapCollectionRec"]
energies = [50]
bounds_software_on = {
    11: {  # e
        15: {50: 0.00562556},
        85: {50: 0.03639618},
    },
    211: {   # p
        15: {50: 0.00410849},
        85: {50: 0.01243161},
    }
}
bounds_software_off = {
    211: {
        15: {50:0.00404678},
        85: {50:0.01238617},
    },
    11: {
        15: {50:0.00534034},
        85: {50:0.03894222},
    }
}
file = uproot.open
angle = [85]
particle = [11]
for pid in particle: 
    for num in energies:
        for a in angle:
            files = [
                f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{num}_theta_{a}-{a}_calib10/reco_chunk_1.root",
                f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{num}_theta_{a}-{a}_calib10/reco_chunk_2.root",
                f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{num}_theta_{a}-{a}_calib10/reco_chunk_3.root",
                f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{num}_theta_{a}-{a}_calib10/reco_chunk_4.root",
            ]
            bound = bounds_software_off[pid][a][num]
            matched_clusters = []
            for f in files: 
                print(f"\n=== Energy {num} GeV ===")
                with uproot.open(f) as file:
                    #file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_2112_pt_{num}_theta_15-15_bib2/reco_pdg_2112_pt_{num}_theta_15-15_bib.root")
                    events = file["events"]
                    pandora_clusters = events["PandoraClusters"]
                    #I need the theta phi of mc and of clusters 
                    mcparticles = events["MCParticles"]
                    #we want a status of 1 
                    status_mc = mcparticles["MCParticles.generatorStatus"].array()
                    #do I want vertex or endpoint? I presume endpoint and we will see both
                    #I will try it first with endpoint
                    mc_x = mcparticles["MCParticles.endpoint.x"].array()
                    mc_y = mcparticles["MCParticles.endpoint.y"].array()
                    mc_z = mcparticles["MCParticles.endpoint.z"].array()
                    mc_momx = mcparticles["MCParticles.momentum.x"].array()
                    mc_momy = mcparticles["MCParticles.momentum.y"].array()
                    mc_momz = mcparticles["MCParticles.momentum.z"].array()
                    mc_mass = mcparticles["MCParticles.mass"].array()
                    #Now I want cluster energy
                    #Also I want cluster theta, phi
                    cluster_x_bib=pandora_clusters["PandoraClusters.position.x"].array()
                    cluster_y_bib=pandora_clusters["PandoraClusters.position.y"].array()
                    cluster_z_bib=pandora_clusters["PandoraClusters.position.z"].array()
                    cluster_energy_bib=pandora_clusters["PandoraClusters.energy"].array()
                    angular_dist = []
                    regular_dist = []
                    pandora_clusters_hits = events["_PandoraClusters_hits"]
                    hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
                    collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
                    hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
                    hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()
                    #Let's right now just filter
                    for i in range((events.num_entries)):
                        ecal_barrel = 0
                        hcal_barrel = 0
                        ecal_endcap = 0
                        hcal_endcap = 0
                        if (i % 100 == 0):
                            print(i)
                        #Let's just right now filter for events with only 1 cluster
                        #if len(cluster_energy[i]) == 1:
                        if True: 
                            #num_events_1_clus +=1 
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
                            #Store cluster energy if passes
                            clus_array = []
                            clus_indices = []
                            #Ok it's no longer going to work for just 1 cluster
                            for j in range(len(cluster_energy_bib[i])):
                                cx = cluster_x_bib[i][j] 
                                cy = cluster_y_bib[i][j]
                                cz = cluster_z_bib[i][j]
                                cenergy = cluster_energy_bib[i][j]
                                c_r = np.sqrt(cx**2 + cy**2 + cz**2)
                                c_theta = np.arccos(cz / c_r)
                                c_phi = np.arctan2(cy, cx)
                                cosang = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta)) #Distance from mc particle
                                #Now we need to clip it
                                cosang = np.clip(cosang, -1.0, 1.0)
                                angular_distance = np.arccos(cosang)  
                                if angular_distance <= bound:
                                    clus_array.append(cenergy)
                                    clus_indices.append(j)
                            if len(clus_array) == 0:
                                continue
                            clus_array = np.array(clus_array)
                            max_energy = np.max(clus_array)
                            mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                            mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                            matched_local = np.argmax(clus_array)
                            matched_index = clus_indices[matched_local]

                            #Now I will be getting ecal_hits and hcal_hits thing
                            hits_begin_arr = hits_begin_all[i]
                            hits_end_arr = hits_end_all[i]
                            hit_index = hit_index_all[i]
                            collection_ID = collectionID_all[i]
                            lo = hits_begin_arr[matched_index]
                            hi = hits_end_arr[matched_index]
                            idxs = hit_index[lo:hi]
                            sysIDs = collection_ID[lo:hi]
                            for code in sysIDs:
                                if code == 679272617:
                                    #print ("hi")
                                    ecal_barrel +=1
                                if code == 1573202488: 
                                    #print("hi")
                                    hcal_barrel +=1
                                    #print ("hii")
                                if code == 3383333369:
                                    ecal_endcap +=1
                                    #print ("hiii")
                                if code == 2381985645:
                                    hcal_endcap +=1
                                    #print ("hiiii")
                            matched_clusters.append({
                                "event": i,
                                "pid": pid,
                                "energy": num,
                                "angle": a,
                                "matched_energy": max_energy,
                                "mc_energy": mc_energy,
                                "ecal_barrel": ecal_barrel,
                                "hcal_barrel": hcal_barrel,
                                "ecal_endcap": ecal_endcap,
                                "hcal_endcap": hcal_endcap,
                            })
                            
                            #Want to store array of event, and matched cluster, with whether its software, no software, degrees, pt
                            #I just want it to pick out the matched clusters
                                # ==========================================
                    # Write matched cluster information to ROOT
                    # ==========================================

            if len(matched_clusters) == 0:
                print("No matches found — skipping file write")
                continue
            outfile = f"matched_clusters_pid{pid}_E{num}_theta{a}_calib10.root"
            output_data = {
                "event": np.array([x["event"] for x in matched_clusters], dtype=np.int32),
                "pid": np.array([x["pid"] for x in matched_clusters], dtype=np.int32),
                "beam_energy": np.array([x["energy"] for x in matched_clusters], dtype=np.float64),
                "angle": np.array([x["angle"] for x in matched_clusters], dtype=np.float64),
                "matched_energy": np.array([x["matched_energy"] for x in matched_clusters], dtype=np.float64),
                "mc_energy": np.array([x["mc_energy"] for x in matched_clusters], dtype=np.float64),
                "ecal_barrel": np.array([x["ecal_barrel"] for x in matched_clusters], dtype=np.int32),
                "hcal_barrel": np.array([x["hcal_barrel"] for x in matched_clusters], dtype=np.int32),
                "ecal_endcap": np.array([x["ecal_endcap"] for x in matched_clusters], dtype=np.int32),
                "hcal_endcap": np.array([x["hcal_endcap"] for x in matched_clusters], dtype=np.int32),
            }

            with uproot.recreate(outfile) as fout:
                fout["MatchedClusters"] = output_data
            
            print(f"Wrote {len(matched_clusters)} matched clusters to {outfile}")