#This will be checking energy ratio here
#We will compare if you add the energy in the 4 instances if this gets the total energy

#Combine the response graphs w/ the ratio graphs 
#Plot the response color code by ratio yellow = one, blue = 0 

#all in hcal = 0
#all in ecal = 1
#So the number is the count in the ecal
#We want to fill an array with response
#Fill another array with the ecal ratio

#This a test of correlating fraction to where fraction of hits are found

#This is plotting the original and the edited functions

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
energies = [50]
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
e = 50
# e 2, 10, 30, 50 
#particle = [2112, 2212, 321]
#particle = [2112]
particle = [211, 11]
angles = [15, 85]
energy = [10, 30, 50]
#angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
yes = True
#This is exluding bounds!
bas_median = []
bas_low = []
bas_high = []
cal_median = []
cal_low = []
cal_high = []
for pid in particle:
    mc_median= []
    mc_low = []
    mc_high = []
    for a in angles:
        for e in energy: 
            print(f"\n=== Energy {a} Degrees ===")
            fraction = []
            energy_dif = []
            num_events_1_clus = 0
            passes = 0
            #bound = bounds[pid][a][10] #this is not a trial run, og run
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial10.2/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial/reco_pdg_pdg_pt_pt_theta_theta_nobib.root")
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_base/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_50_theta_{a}-{a}_bib2/reco_pdg_{pid}_pt_50_theta_{a}-{a}_nobib.root")
            #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_basesoft/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
            file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_{e}_theta_{a}-{a}_bib2/job_0/reco_output_p{e}_{pid}_nobib0.edm4hep.root")
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
            ecal_ratio = []
            hcal_ratio = []
            for i in range((events.num_entries)):
                
                #Let's just right now filter for events with only 1 cluster
                ecal_endcap = 0
                ecal_barrel = 0
                hcal_endcap = 0
                hcal_barrel = 0
                ecal_total = 0
                total = 0
                if len(cluster_energy[i]) == 1:
                    array = np.array(pandora_sub_energy[i])
                    ecal_energy = array[0]
                    hcal_energy = array[1]
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
                        clus_energy = cluster_energy[i][0]
                        fraction_mc = cluster_energy[i][0] / mc_energy[0]
                        total_energy = ecal_energy + hcal_energy
                        energy_difference = (clus_energy - total_energy) / clus_energy
                        energy_dif.append(energy_difference)
                        ecal_ratio.append(ecal_energy / total_energy)
                        hcal_ratio.append (hcal_energy / total_energy)
                        fraction.append(fraction_mc)        
            fraction = np.array(fraction) #cluster / mx[0] to recalibrate multiply the top number 
            energy_dif = np.array(energy_dif)
            plt.hist(energy_dif, bins=50)
            plt.xlabel("Energy Difference")
            plt.ylabel("Counts")
            plt.title(f"Percent Difference Ecal/Hcal Energy vs Cluster Energy {pid} {a} {e} w/ Soft")
            plt.savefig(f"energy_vs_ecal_hcal{pid}_{a}_{e}_withsoft.pdf")
            plt.close()
