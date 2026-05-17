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
angles = [140, 141, 142, 143, 144, 145, 146]
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
particle = [11]
yes = True
for pid in particle:
    mc_median= []
    mc_low = []
    mc_high = []
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")
        fraction = []
        num_events_1_clus = 0
        passes = 0
        #bound = bounds[pid][a][10] #this is not a trial run, og run
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_{e}_theta_{a}-{a}_trial5/reco_pdg_pdg_pt_pt_theta_theta_nobib.root")
        events = file["events"]
        pandora_clusters = events["PandoraClusters"]
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
        for i in range((events.num_entries)):
            #Let's just right now filter for events with only 1 cluster
            ecal_endcap = 0
            ecal_barrel = 0
            hcal_endcap = 0
            hcal_barrel = 0
            if len(cluster_energy[i]) == 1:
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
                    print("hi")
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
                                hcal_barrel +=1
                                #print ("hii")
                            if code == 3383333369:
                                ecal_endcap +=1
                                #print ("hiii")
                            if code == 2381985645:
                                hcal_endcap +=1
                                #print ("hiiii")
                    total = ecal_barrel + hcal_barrel + ecal_endcap + hcal_endcap
                    #print(total)
                    #print(ecal_barrel)
                    #print(ecal_endcap)
                    #print(hcal_barrel)
                    #print(hcal_endcap)
                    #print(ecal_barrel / total)
                    #print(ecal_endcap / total)
                    #print(hcal_barrel / total)
                    #print(hcal_endcap / total)
                    ecal_e_ratio.append(ecal_endcap / total)
                    hcal_e_ratio.append(hcal_endcap / total)
                    ecal_b_ratio.append(ecal_barrel / total)
                    hcal_b_ratio.append(hcal_barrel / total)
        fraction = np.array(fraction)
        ecal_b_ratio = np.array(ecal_b_ratio) #Is this still allowed if its 0 
        ecal_e_ratio = np.array(ecal_e_ratio)
        hcal_b_ratio = np.array(hcal_b_ratio)
        hcal_e_ratio = np.array(hcal_e_ratio)
        #scatter plot of fraction and each ratio together
        # Create scatter plot
        plt.figure(figsize=(8,6))
        plt.scatter(fraction, ecal_b_ratio, label='Ecal Barrel', alpha=0.7)
        plt.xlabel('Response')
        plt.ylabel('Ratio of Hits')
        plt.title(f'Response vs Detector Ratios for {pid} and {a} angle {e} pt')
        plt.legend()
        plt.grid()
        plt.savefig(f"scatter_plot_pid{pid}_angle{a}_{e}_ecalb.png")
        plt.close()

        plt.scatter(fraction, ecal_e_ratio, label='Ecal Endcap', alpha=0.7)
        plt.xlabel('Response')
        plt.ylabel('Ratio of Hits')
        plt.title(f'Response vs Detector Ratios for {pid} and {a} angle {e} pt')
        plt.legend()
        plt.grid()
        plt.savefig(f"scatter_plot_pid{pid}_angle{a}_{e}_ecale.png")
        plt.close()

        plt.scatter(fraction, hcal_b_ratio, label='Hcal Barrel', alpha=0.7)
        plt.xlabel('Response')
        plt.ylabel('Ratio of Hits')
        plt.title(f'Response vs Detector Ratios for {pid} and {a} angle {e} pt')
        plt.legend()
        plt.grid()
        plt.savefig(f"scatter_plot_pid{pid}_angle{a}_{e}_hcalb.png")
        plt.close()

        plt.scatter(fraction, hcal_e_ratio, label='Hcal Endcap', alpha=0.7)
        plt.xlabel('Response')
        plt.ylabel('Ratio of Hits')
        plt.title(f'Response vs Detector Ratios for {pid} and {a} angle {e} pt')
        plt.legend()
        plt.grid()
        plt.savefig(f"scatter_plot_pid{pid}_angle{a}_{e}_hcale.png")
        plt.close()
        


#Ok right now mc_median stores the baseline energies 
#Now I want to look at trials
'''
for pid in particle:
    mc_trial= []
    mc_tlow = []
    mc_thigh = []
    for a in angles:
        print(f"\n=== Angle {a} Degrees ===")
        fraction = []
        num_events_1_clus = 0
        passes = 0
        bound = bounds[pid][a][50]
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_50_theta_{a}-{a}_trial1/reco_pdg_{pid}_pt_50_theta_{a}-{a}_nobib.root")
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
        cluster_x=pandora_clusters["PandoraClusters.position.x"].array()
        cluster_y=pandora_clusters["PandoraClusters.position.y"].array()
        cluster_z=pandora_clusters["PandoraClusters.position.z"].array()
        cluster_energy=pandora_clusters["PandoraClusters.energy"].array()
        angular_dist = []
        regular_dist = []
        #Let's right now just filter
        #for i in range(events.num_entries):
        #for i in range((5001)):
        for i in range((events.num_entries)):
            #Let's just right now filter for events with only 1 cluster
            if len(cluster_energy[i]) == 1:
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
                if angular_distance <= bound:
                    passes += 1
                    mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                    mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                    fraction_mc = cluster_energy[i][0] / mc_energy[0]
                    fraction.append(fraction_mc)
        fraction = np.array(fraction)
        q16, q84 = np.percentile(fraction, [16, 84])
        median = np.median(fraction)
        mc_tlow.append(median-q16)
        mc_thigh.append(q84-median)
        mc_trial.append(median)
        #But we also want to graph displays
        #bins=30
        bins = np.linspace(.6, 1.3, 30)  # 60 bins → need 61 edges
        plt.xlim(0.6, 1.3)
        plt.hist(fraction, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Cluster/MC Energy Fraction")
        plt.ylabel("Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.6f}"
        )
        plt.legend()
        plt.title(f"Trial1 {a} deg given 1 Cluster PID: {pid} Nobib Energy 50 GeV")
        plt.tight_layout()
        plt.savefig(f"trial1_{a}_fraction_nobib_{pid}50GeV.pdf")
        plt.close()
    
    #Summary graph, number of bib clusters given 1 normal cluster
    plt.errorbar(angles, mc_median, yerr=[mc_low, mc_high], fmt='s', alpha= 0.6, capsize=4, label=f"Baseline {pid}")
    plt.errorbar(angles, mc_trial, yerr=[mc_tlow, mc_thigh], fmt='s', alpha= 0.6, capsize=4, label=f"Trial1 {pid}")
    plt.xlabel("Particle gun angle (degrees)")
    plt.ylabel("Median Cluster/MC Energy")
    plt.title(f" 50 GeV Median Cluster/MC Energy 1 Nobib Cluster Pid: {pid} ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"trial1_regular_{pid}.pdf")
    plt.close()
'''