#Here we're going to look at the number of hits in general, then number of hits before bib, how much gets added w/ bib

#Let's do it first the nobib cluster lists -> loop

angles = [15, 85]
particle = [211, 11]
energy = [50]

#/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_15-15_bib2/job_0
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
region_colors = {
    "Barrel": "tab:orange",
    "Endcap": "tab:blue",
}

calib_styles = {
    "Baseline": "-",
    "4Cal": "--",
    "4CalHitCal": ":",
}

calib_markers = {
    "Baseline": "o",   # circle
    "4Cal": "s",      # square
    "4CalHitCal": "^" # triangle
}

particle_labels = {
    11: "Electron",
    211: "Pion"
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
energy = [10,30,50]
yes = True
results = {}
results_base = {}
results_4cal = {}
#This is exluding bounds!
angle = [85]
particle = [211]
energy = [50]

for pid in particle:
    for e in energy: 
        mc_median= []
        mc_low = []
        mc_high = []
        for a in angles:
            print(f"\n=== Energy {a} Degrees ===")
            fraction = []
            num_events_1_clus = 0
            passes = 0
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_bib2/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_85-85_clustertest/reco_pdg_211_pt_50_theta_85-85_nobib.root")
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
            
            num_hits_arrays = []
            for i in range((events.num_entries)):
                #Let's just right now filter for events with only 1 cluster
                ecal_endcap = 0
                ecal_barrel = 0
                hcal_endcap = 0
                hcal_barrel = 0
                ecal_total = 0
                total = 0
                if yes:
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
                            num_hits = hi - lo
                            num_hits_arrays.append(num_hits)
            #Now we are going to make a gaussian plot for this
            num_hits_arrays = np.array(num_hits_arrays)
            q16, q84 = np.percentile(num_hits_arrays, [16, 84])
            bins = 100
            plt.figure(figsize=(7,5))
            median = np.median(num_hits_arrays)
            plt.hist(num_hits_arrays, bins=bins, alpha=1, label=(f'Number of Hits per No-Bib Cluster {pid}, {e}, {a}'))
            plt.axvline(median, color='red', linestyle='--', linewidth=2, label=f'Median = {median}')
            plt.axvline(q16, color='royalblue', linestyle='--', linewidth=2, label=f'16th Percentile = {q16}')
            plt.axvline(q84, color='green', linestyle='--', linewidth=2, label=f'84th Percentile = {q84}')
            plt.legend()
            plt.savefig(f"numhits_nobib_{pid}_{e}_{a}_clustertest.png")
            plt.close()

                            