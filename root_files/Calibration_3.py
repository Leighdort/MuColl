#3 response graphs on one graphs


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
angles = [15, 85, 140]
'''
bounds = {
    11: {   # electron 85 -85 
        2: 1.40, 
        10: 0.09,
        30: 0.06,
        50: 0.06,
    },
    11: { #electron 143-143
        2: 0.59,
        5: 0.05,
        10: 0.04,
        15: 0.04,
        30: 0.04,
        50: 0.04
    },
    211: {  # pion 143-143
        2: 0.41,
        10: 0.03,
        30: 0.01,
        50: 0.008
    },
    211: { #pion 85 -85 
        2: 0.82,
        10: 0.06,
        30: 0.02,
        50: 0.02
    },
    211: { # pion 15-15
        2: 0.05,
        10: 0.02,
        30: 0.01,
        50: 0.006,
    },
    11: {# electron 15-15
        2: 0.01,
        10: 0.01,
        30: 0.01,
        50: 0.01,
    }
}
'''
bounds = {
    211: {  # pion
        15: {2: 0.05, 10: 0.02, 30: 0.01, 50: 0.006},
        85: {2: 0.82, 10: 0.06, 30: 0.02, 50: 0.02},
        140:{2: 0.41, 10: 0.03, 30: 0.01, 50: 0.008},
    },
    11: {   # electron
        15: {2: 0.01, 10: 0.01, 30: 0.01, 50: 0.01},
        85: {2: 1.40, 10: 0.09, 30: 0.06, 50: 0.06},
        140:{2: 0.59, 5: 0.05, 10: 0.04, 15: 0.04, 30: 0.04, 50: 0.04},
    }
}
yes = True 
particle = [211]
#YOU MAY UNDER NO CIRCUMSTANCES WRITE MORE THAN ONE PARTICLE THERE
#THIS IS HARD CODED

for pid in particle:
    mc_median= []
    mc_low = []
    mc_high = []
    cal = []
    cal_low = []
    cal_high = []
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")
        fraction = []
        num_events_1_clus = 0
        passes = 0
        #bound = bounds[pid][a][50]
        #a_pdg_211_pt_50_theta_140-140_trial6/job_0/reco_output_p50_211_nobib0.edm4hep.root
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_base/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_50_theta_{a}-{a}_bib2/reco_pdg_211_pt_50_theta_{a}-{a}_nobib.root")
        print(len(file))
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
        print(events.num_entries)
        for i in range((5000)):
        #for i in range(events.num_entries):
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
                if yes:
                    passes += 1
                    mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                    mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                    fraction_mc = cluster_energy[i][0] / mc_energy[0]
                    fraction.append(fraction_mc)
        fraction = np.array(fraction)
        q16, q84 = np.percentile(fraction, [16, 84])
        median = np.median(fraction)
        mc_low.append(median-q16)
        mc_high.append(q84-median)
        mc_median.append(median)
        #bins = 30
        bins = np.linspace(.6, 1.4, 30)  # 60 bins → need 61 edges
        plt.xlim(0.6, 1.4)
        plt.hist(fraction, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Response")
        plt.ylabel("Count")
        plt.axvline(
            median,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median:.6f}"
        )
        plt.legend()
        plt.title(f"Original {a} deg Response PID: {pid} Nobib Energy 50 GeV")
        plt.tight_layout()
        plt.savefig(f"norm10_{a}_fraction_nobib_{pid}50GeV.pdf")
        plt.close()
#Mc_median now carries 1 value per angle

#Ok right now mc_median stores the baseline energies 
#Now I want to look at trials
#/users/rldohert/data/mucoll/rldohert/a_pdg_211_pt_50_theta_85-85_trial6/job_0/reco_output_p50_pt_nobib0.edm4hep.root
#/users/rldohert/data/mucoll/rldohert/a_pdg_211_pt_50_theta_85-85_trial6/job_0/reco_output_p50_211_nobib0.edm4hep.root
for pid in particle:
    mc_trial= []
    mc_tlow = []
    mc_thigh = []
    cal = []
    cal_high = []
    cal_low = []
    for a in angles:
        print(f"\n=== Angle {a} Degrees ===")
        fraction = []
        num_events_1_clus = 0
        passes = 0
        #bound = bounds[pid][a][50]
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_50_theta_{a}-{a}_trial5/reco_pdg_{pid}_pt_50_theta_{a}-{a}_nobib.root")
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial10/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial8/reco_pdg_pdg_pt_pt_theta_theta_nobib.root")
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
        #Let's right now just filter
        #for i in range(events.num_entries):
        #for i in range((5001)):
        print("meow")
        print(events.num_entries)
        pandora_clusters = events["PandoraClusters"]
        #I need the theta phi of mc and of clusters 
        mcparticles = events["MCParticles"]
        #we want a status of 1 
        status_mc = mcparticles["MCParticles.generatorStatus"].array()
        #do I want vertex or endpoint? I presume endpoint and we will see both
        #I will try it first with endpoint
        pandora_clusters_hits = events["_PandoraClusters_hits"]
        collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
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
        for i in range((5000)):
        #for i in range(events.num_entries):
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
                #if angular_distance <= bound:
                ecal_endcap = 0
                ecal_barrel = 0
                hcal_endcap = 0
                hcal_barrel = 0
                ecal_total = 0
                total = 0
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
        fraction = np.array(fraction) #cluster / mx[0] to recalibrate multiply the top number 
        ecal_ratio = np.array(ecal_ratio)
        ecal_e_ratio = np.array(ecal_e_ratio)
        ecal_b_ratio = np.array(ecal_b_ratio)
        hcal_e_ratio = np.array(hcal_e_ratio)
        hcal_b_ratio = np.array(hcal_b_ratio)
        barrel = ecal_b_ratio + hcal_b_ratio
        endcap = ecal_e_ratio + hcal_e_ratio
        fraction = np.array(fraction)
        q16, q84 = np.percentile(fraction, [16, 84])
        median = np.median(fraction)
        mc_tlow.append(median-q16)
        mc_thigh.append(q84-median)
        mc_trial.append(median)
        #But we also want to graph displays
        #bins=30
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_e_ratio[i] * .09
            correction_2 =  hcal_e_ratio[i] * .04 
            correction_3 = ecal_b_ratio[i] * .16
            correction_4 =  hcal_b_ratio[i] * .19
            new_fraction = baseline - correction_1 + correction_2 - correction_3 + correction_4
            fraction[i] = new_fraction
        median = np.median(fraction)
        q16, q84 = np.percentile(fraction, [16, 84])
        cal_low.append(median-q16)
        cal_high.append(q84-median)
        cal.append(median)
new_low_211  = [0.0402340210299903, 0.07934760302849009, 0.09779884682419804]
new_median_211 = [0.9892366576484928, 1.0068692863917286, 0.9682049822457335]
new_high_211 = [0.03633895811819721, 0.0693490172023623, 0.06323305104923094]
new_low_2112 = [0.0359713446280856, 0.06956301920225239, 0.10582252399464209]
new_median_2112 = [0.9838472017157688,0.9878281573647439,0.9520988330067548]
new_high_2112 = [0.03437333233335005,0.06526503636708736,0.06758832267421422]
plt.errorbar(angles, mc_median, yerr=[mc_low, mc_high], fmt='s', alpha= 0.6, capsize=4, label=f"Baseline {pid}")
plt.errorbar(angles, mc_trial, yerr=[mc_tlow, mc_thigh], fmt='s', alpha= 0.6, capsize=4, label=f"Trial10  {pid}")
#plt.errorbar(angles, cal, yerr=[cal_low, cal_high], fmt='s', alpha= 0.6, capsize=4, label=f"Hit Calibration {pid}")
plt.errorbar(angles, new_median_211, yerr = [new_low_211, new_high_211], fmt='s', alpha = 0.6, capsize = 4, label=f"Energy Calibration {pid}")
plt.xlabel("Particle gun angle (degrees)")
plt.ylabel("Median Response")
plt.title(f" Trial10: 50 GeV Response Pid: {pid} ")
#plt.title(f" Original 50 GeV Median Response 1 Nobib Cluster Pid: {pid} ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"trial10_new_{pid}.pdf")
plt.close()

