#This is just doing Gaussian_res.py for base w/ a set number of events (5000)
#May 17, 2026
#This gives you gaussians

#Combine the response graphs w/ the ratio graphs 
#Plot the response color code by ratio yellow = one, blue = 0 

#all in hcal = 0
#all in ecal = 1
#So the number is the count in the ecal
#We want to fill an array with response
#Fill another array with the ecal ratio

#This a test of correlating fraction to where fraction of hits are found

#This is plotting the original and the edited functions
angle2region = {
    15: "Endcap",
    85: "Barrel",
    143: "Transition",
}

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
energy = [10, 30, 50]
#energy = [30, 50]
# e 2, 10, 30, 50 
#particle = [2112, 2212, 321]
particle = [11, 211]
#angles = [15, 85, 140]
angles = [15, 85]
yes = True
results = {}
results_base = {}
results_4cal = {}
nosoft_median = {}
nosoft_low = {}
nosoft_high = {}
soft_median = {}
soft_low = {}
soft_high = {}
#This is exluding bounds!
for pid in particle:
    median_soft = {15: [], 85: []}
    low_soft    = {15: [], 85: []}
    high_soft   = {15: [], 85: []}
    median_nosoft = {15: [], 85: []}
    low_nosoft    = {15: [], 85: []}
    high_nosoft   = {15: [], 85: []}
    res_soft = {15: [], 85: []}
    res_nosoft = {15: [], 85: []}
    for e in energy: 
        response = {}
        for a in angles:
            print(f"\n=== Energy {e} {a} Degrees ===")
            fraction = []
            num_events_1_clus = 0
            passes = 0
            file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_bib2/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_basesoft/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            events = file["events"]
            pandora_clusters = events["PandoraClusters"]
            pandora_sub_energy = events["_PandoraClusters_subdetectorEnergies"].array()
            pandora_sub_begin = pandora_clusters["PandoraClusters.subdetectorEnergies_begin"].array()
            pandora_sub_end = pandora_clusters["PandoraClusters.subdetectorEnergies_end"].array()
            
            #I need the theta phi of mc and of clusters 
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
            ecal_e_ratio = []
            hcal_e_ratio = []
            ecal_b_ratio = []
            hcal_b_ratio = []
            ecal_ratio = []
            hcal_ratio = []
            ecal_energy_ratio = []
            hcal_energy_ratio = []
            for i in range((5000)):
                #Let's just right now filter for events with only 1 cluster
                ecal_endcap = 0
                ecal_barrel = 0
                hcal_endcap = 0
                hcal_barrel = 0
                ecal_total = 0
                total = 0
                if len(cluster_energy[i]) == 1:
                    array = np.array(pandora_sub_energy[i])
                    num_events_1_clus +=1 
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
                    #distance = np.sqrt(mc_r**2 + c_r**2 * (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi-c_phi) + np.cos(c_theta)*np.cos(mc_theta)))
                    #Now we can try to do angular distance
                    cosang = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                    #Now we need to clip it
                    cosang = np.clip(cosang, -1.0, 1.0)
                    angular_distance = np.arccos(cosang)
                    #This is important because this tells us the angle
                    #Angle and making sure momentum is in the same direction is more important
                    other_angle = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                    other_dist = np.sqrt(mc_r**2 + c_r**2 - 2*mc_r*c_r*other_angle)
                    #We will be using angular_distance
                    #if angular_distance <= bound:
                    if yes == True: 
                        #print("hi")
                        passes += 1
                        mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                        mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                        fraction_mc = cluster_energy[i][0] / mc_energy[0]
                        fraction.append(fraction_mc)
                        hits_begin_arr = hits_begin_all[i]
                        hits_end_arr = hits_end_all[i]
                        collection_ID = collectionID_all[i]
                        for j in range(len(hits_begin_arr)): #this length should only be 1
                            #print(len(hits_begin_arr)) #comment this out later
                            lo = hits_begin_arr[j]
                            hi = hits_end_arr[j]
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
                        total = ecal_barrel + hcal_barrel + ecal_endcap + hcal_endcap
                        ecal_e_ratio.append(ecal_endcap / total)
                        hcal_e_ratio.append(hcal_endcap / total)
                        ecal_b_ratio.append(ecal_barrel / total)
                        hcal_b_ratio.append(hcal_barrel / total)
                        ecal_total = ecal_barrel + ecal_endcap
                        ecal_ratio.append(ecal_total / total)
                        hcal_total = hcal_barrel + hcal_endcap
                        hcal_ratio.append(hcal_total / total )
                        ecal_sub_energy = array[0]
                        hcal_sub_energy = array[1]
                        tots = ecal_sub_energy + hcal_sub_energy
                        ecal_energy_ratio.append(ecal_sub_energy/tots)
                        hcal_energy_ratio.append(hcal_sub_energy/tots)
            fraction = np.array(fraction) #cluster / mx[0] to recalibrate multiply the top number 
            q16, q84 = np.percentile(fraction, [16, 84])
            median_full = np.median(fraction)
            median_soft[a].append(median_full)
            low_soft[a].append(median_full - q16)
            high_soft[a].append(q84 - median_full)
            resolution = (q84 - q16)/(2*median_full)
            res_soft[a].append(resolution)

            print(f"\n=== Energy Nosoft {e} {a} Degrees ===")
            fraction = []
            num_events_1_clus = 0
            passes = 0
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_bib2/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_basesoft/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            events = file["events"]
            pandora_clusters = events["PandoraClusters"]
            pandora_sub_energy = events["_PandoraClusters_subdetectorEnergies"].array()
            pandora_sub_begin = pandora_clusters["PandoraClusters.subdetectorEnergies_begin"].array()
            pandora_sub_end = pandora_clusters["PandoraClusters.subdetectorEnergies_end"].array()
            
            #I need the theta phi of mc and of clusters 
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
            ecal_e_ratio = []
            hcal_e_ratio = []
            ecal_b_ratio = []
            hcal_b_ratio = []
            ecal_ratio = []
            hcal_ratio = []
            ecal_energy_ratio = []
            hcal_energy_ratio = []
            for i in range((5000)):
                #Let's just right now filter for events with only 1 cluster
                ecal_endcap = 0
                ecal_barrel = 0
                hcal_endcap = 0
                hcal_barrel = 0
                ecal_total = 0
                total = 0
                if len(cluster_energy[i]) == 1:
                    array = np.array(pandora_sub_energy[i])
                    num_events_1_clus +=1 
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
                    #distance = np.sqrt(mc_r**2 + c_r**2 * (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi-c_phi) + np.cos(c_theta)*np.cos(mc_theta)))
                    #Now we can try to do angular distance
                    cosang = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                    #Now we need to clip it
                    cosang = np.clip(cosang, -1.0, 1.0)
                    angular_distance = np.arccos(cosang)
                    #This is important because this tells us the angle
                    #Angle and making sure momentum is in the same direction is more important
                    other_angle = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                    other_dist = np.sqrt(mc_r**2 + c_r**2 - 2*mc_r*c_r*other_angle)
                    #We will be using angular_distance
                    #if angular_distance <= bound:
                    if yes == True: 
                        #print("hi")
                        passes += 1
                        mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                        mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                        fraction_mc = cluster_energy[i][0] / mc_energy[0]
                        fraction.append(fraction_mc)
                        hits_begin_arr = hits_begin_all[i]
                        hits_end_arr = hits_end_all[i]
                        collection_ID = collectionID_all[i]
                        for j in range(len(hits_begin_arr)): #this length should only be 1
                            #print(len(hits_begin_arr)) #comment this out later
                            lo = hits_begin_arr[j]
                            hi = hits_end_arr[j]
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
                        total = ecal_barrel + hcal_barrel + ecal_endcap + hcal_endcap
                        ecal_e_ratio.append(ecal_endcap / total)
                        hcal_e_ratio.append(hcal_endcap / total)
                        ecal_b_ratio.append(ecal_barrel / total)
                        hcal_b_ratio.append(hcal_barrel / total)
                        ecal_total = ecal_barrel + ecal_endcap
                        ecal_ratio.append(ecal_total / total)
                        hcal_total = hcal_barrel + hcal_endcap
                        hcal_ratio.append(hcal_total / total )
                        ecal_sub_energy = array[0]
                        hcal_sub_energy = array[1]
                        tots = ecal_sub_energy + hcal_sub_energy
                        ecal_energy_ratio.append(ecal_sub_energy/tots)
                        hcal_energy_ratio.append(hcal_sub_energy/tots)
            fraction = np.array(fraction) #cluster / mx[0] to recalibrate multiply the top number 
            q16, q84 = np.percentile(fraction, [16, 84])
            median_full = np.median(fraction)
            median_nosoft[a].append(median_full)
            low_nosoft[a].append(median_full - q16)
            high_nosoft[a].append(q84 - median_full)
            resolution = (q84 - q16)/(2*median_full)
            res_nosoft[a].append(resolution)
    #Summery plot for 11 or 211 
            
    #Here we make summery plots
    for a in angles:
        plt.errorbar(
            energy,
            median_nosoft[a],
            yerr=[low_nosoft[a], high_nosoft[a]],
            fmt='o-',
            capsize=4,
            label=f"{a}° Software Off"
        )
        plt.errorbar(
            energy,
            median_soft[a],
            yerr=[low_soft[a], high_soft[a]],
            fmt='s--',
            capsize=4,
            label=f"{a}° Software On"
        )
    plt.xlabel("Beam Energy (GeV)")
    plt.ylabel("Median Response")
    plt.title(f"One Cluster Response with Software On and Off for {pid}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"one_cluster_response_softonoff{pid}.pdf")
    plt.close()

    for a in angles:

        plt.plot(
            energy,
            res_soft[a],
            's--',
            linewidth=2,
            label=f"{a}° Software On"
        )
    
        plt.plot(
            energy,
            res_nosoft[a],
            'o-',
            linewidth=2,
            label=f"{a}° Software Off"
        )
    
    plt.xlabel("Beam Energy (GeV)")
    plt.ylabel("Resolution")
    plt.title(f"One Cluster Resolution with Software On and Off for PID {pid}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"one_cluster_resolution_softonoff{pid}.pdf")
    plt.close()



