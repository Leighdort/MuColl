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
particle = [211]
#angles = [15, 85, 140]
angles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
yes = True
#This is exluding bounds!
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
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial10.2/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial5/reco_pdg_pdg_pt_pt_theta_theta_nobib.root")
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_base/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_50_theta_{a}-{a}_bib2/reco_pdg_{pid}_pt_50_theta_{a}-{a}_nobib.root")
        #file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_{pid}_pt_50_theta_{a}-{a}_bib2/reco_pdg_{pid}_pt_50_theta_{a}-{a}_nobib.root")
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
        fraction = np.array(fraction) #cluster / mx[0] to recalibrate multiply the top number 
        ecal_ratio = np.array(ecal_ratio)
        hcal_ratio = np.array(hcal_ratio)
        ecal_e_ratio = np.array(ecal_e_ratio)
        ecal_b_ratio = np.array(ecal_b_ratio)
        hcal_e_ratio = np.array(hcal_e_ratio)
        hcal_b_ratio = np.array(hcal_b_ratio)
        barrel = ecal_b_ratio + hcal_b_ratio
        endcap = ecal_e_ratio + hcal_e_ratio
        #Attempt one at calibrations!
        '''
        for i in range(len(fraction)):
            if ecal_ratio[i] > 0.5:
                if ecal_e_ratio[i] > ecal_b_ratio[i]:
                    fraction[i] *= 0.95
                else:
                    fraction[i] *= 0.90
            else: 
                if hcal_e_ratio[i] > hcal_b_ratio[i]:
                    fraction[i] *= 1.02
                else:
                    fraction[i] *= 1.12
        '''
        #Test 1:
        '''
        for i in range(len(fraction)):
            baseline = fraction[i]
            if endcap[i] > barrel[i]:
                total = ecal_e_ratio[i] + hcal_e_ratio[i]
                correction_1 = (ecal_e_ratio[i] / total) * .05
                correction_2 =  (hcal_e_ratio[i] / total) * .02
                new_fraction = baseline - correction_1 + correction_2
                fraction[i] = new_fraction
            else:
                total = ecal_b_ratio[i] + hcal_b_ratio[i]
                correction_1 = (ecal_b_ratio[i] / total) * .10
                correction_2 =  (hcal_b_ratio[i] / total) * .12
                new_fraction = baseline - correction_1 + correction_2
                fraction[i] = new_fraction
        '''
        '''
        #Test 2
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_e_ratio[i] * .05
            correction_2 =  hcal_e_ratio[i] * .02 
            correction_3 = ecal_b_ratio[i] * .10
            correction_4 =  hcal_b_ratio[i] * .12
            new_fraction = baseline - correction_1 + correction_2 - correction_3 + correction_4
            fraction[i] = new_fraction
        print(len(fraction))
        #Then want to do an ecal_endcap vs ecal_barrel check
        #Let's simply cut on above or 50 for one graph
        #50 for another
        '''
        #Test 3
        '''
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_e_ratio[i] * .07
            correction_2 =  hcal_e_ratio[i] * .03 
            correction_3 = ecal_b_ratio[i] * .13
            correction_4 =  hcal_b_ratio[i] * .17
            new_fraction = baseline - correction_1 + correction_2 - correction_3 + correction_4
            fraction[i] = new_fraction
        print(len(fraction))
        '''
        #Test4
        '''
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_e_ratio[i] * .08
            correction_2 =  hcal_e_ratio[i] * .03 
            correction_3 = ecal_b_ratio[i] * .15
            correction_4 =  hcal_b_ratio[i] * .18
            new_fraction = baseline - correction_1 + correction_2 - correction_3 + correction_4
            fraction[i] = new_fraction
        '''
        '''
        #Test5: THIS HAS BEEN THE CHOSEN ONE 
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_e_ratio[i] * .09
            correction_2 =  hcal_e_ratio[i] * .03 
            correction_3 = ecal_b_ratio[i] * .16
            correction_4 =  hcal_b_ratio[i] * .19
            new_fraction = baseline - correction_1 + correction_2 - correction_3 + correction_4
            fraction[i] = new_fraction
        '''
        #This test we're just assuming we only have access to ecal or hcal hits
        #This will behave like a switch 
        #Adjustment5
        '''
        if (a >= 320) or (a <= 40) or (140 <= a <= 220): #endcap
            for i in range(len(fraction)):
                baseline = fraction[i]
                correction_1 = ecal_ratio[i] * .09
                correction_2 = hcal_ratio[i] * .03
                new_fraction = baseline - correction_1 + correction_2
                fraction[i] = new_fraction
        else: #barrel
            for i in range(len(fraction)):
                baseline = fraction[i]
                correction_1 = ecal_ratio[i] * .16
                correction_2 = hcal_ratio[i] * .19
                new_fraction = baseline - correction_1 + correction_2
                fraction[i] = new_fraction
        '''
        
        #This one we're trying shift the calibrated This is flaming trash 
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_ratio[i] * 0.13
            correction_2 =  hcal_ratio[i] * .10
            new_fraction = baseline - correction_1 + correction_2
            fraction[i] = new_fraction
        
        '''
        #Testing onto the original line
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_e_ratio[i] * 0.10
            correction_2 =  hcal_e_ratio[i] * .10 
            correction_3 = ecal_b_ratio[i] * .07
            correction_4 =  hcal_b_ratio[i] * .095
            new_fraction = baseline - correction_1 - correction_2 - correction_3 - correction_4
            fraction[i] = new_fraction
        '''
        '''
        #Calibration for just ecal and hcal
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_ratio[i] * 0.085
            correction_2 =  hcal_ratio[i] * .10
            new_fraction = baseline - correction_1 - correction_2
            fraction[i] = new_fraction
        '''
        print(len(fraction))
        #Then want to do an ecal_endcap vs ecal_barrel check
        #Let's simply cut on above or 50 for one graph
        #50 for another
        below_50 = []
        above_50 = []
        ecal_low = []
        ecal_high = []
        #Could try to multiply ratio * the fractions I found
        #Then could just do the cut like if its above 50 shift by this below 50 shift by this 
        for i in range(len(fraction)):
            if ecal_ratio[i] < 0.5:
                below_50.append(fraction[i])
                ecal_low.append(ecal_ratio[i])
            if ecal_ratio[i] >= 0.5:
                above_50.append(fraction[i])
                ecal_high.append(ecal_ratio[i])
        
        bins = np.linspace(0.6, 1.4, 30)
        below_50 = np.array(below_50)
        above_50 = np.array(above_50)
        ecal_low = np.array(ecal_low)
        ecal_high = np.array(ecal_high)
        median_below = np.median(below_50)
        median_above = np.median(above_50)
        '''
        plt.xlim(0.6, 1.4)
        plt.hist(below_50, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Response")
        plt.ylabel("Count")
        plt.axvline(
            median_below,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median_below:.6f}"
        )
        plt.legend()
        plt.title(f"Below 0.50 ecal Ratio {a} {pid} Nobib Energy 50 GeV Ratio")
        plt.tight_layout()
        plt.savefig(f"below_{a}_fraction_nobib_{pid}50GeV_ratio.pdf")
        plt.close()
        plt.hist(above_50, bins=bins, edgecolor = 'black')
        plt.xlabel(f"Response")
        plt.ylabel("Count")
        plt.axvline(
            median_above,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Median = {median_above:.6f}"
        )
        plt.legend()
        plt.title(f"Above 0.50 ecal Ratio {a} {pid} Nobib Energy 50 GeV Ratio")
        plt.tight_layout()
        plt.savefig(f"above_{a}_fraction_nobib_{pid}50GeV_ratio.pdf")
        plt.close()
        '''
        #Can we plot these on the same graph with different opacities adn colors ontop of eachother
        #One labeled below 50
        #One labeled above 50
        bins = np.linspace(0.6, 1.4, 30)
        plt.figure(figsize=(7,5))

        # Below 0.5
        plt.hist(
            below_50,
            bins=bins,
            alpha=0.5,
            label='Ecal ratio < 0.5',
            edgecolor='black'
        )

        # Above 0.5
        plt.hist(
            above_50,
            bins=bins,
            alpha=0.5,
            label='Ecal ratio ≥ 0.5',
            edgecolor='black'
        )

        # Medians
        plt.axvline(
            median_below,
            linestyle='--',
            linewidth=2,
            label=f'Below 0.5 median = {median_below:.3f}'
        )

        plt.axvline(
            median_above,
            linestyle='--',
            linewidth=2,
            label=f'Above 0.5 median = {median_above:.3f}'
        )

        plt.xlabel("Response")
        plt.ylabel("Count")
        plt.xlim(0.6, 1.4)
        plt.title(f"Response ({pid}, {a}°, 50 GeV) Hit Calib 2Cal")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"combined_{a}_fraction_{pid}_hit_2Cal_calibrate.pdf")
        plt.close()
        #Let's just plot fraction now w/ energy

        #Now just a normal graph: 
        q16, q84 = np.percentile(fraction, [16, 84])
        median = np.median(fraction)
        #But we also want to graph displays
        #bins=30
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
        plt.title(f"Response {a} deg PID: {pid} 50 GeV Calib Hit 2Cal")
        plt.tight_layout()
        plt.savefig(f"ratio_{a}_gaussian_{pid}50GeV_hit_2Cal_calibrate.pdf")
        plt.close()
        
        '''
        #We will now plot the average ecal_ratio in each bin
        #I want to overlay it over my normal binning
        #also put error bars on it
        #Lets do 0.6 to 1.4 binning
        bins = np.linspace(0.6, 1.4, 30)
        digitized = np.digitize(below_50, bins) #assigns each thing a bin index 
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_medians = []
        p15 = []
        p85 = []
        for i in range(1, len(bins)): #digitize returns an index starting at 1 
            values = ecal_low[digitized == i]
            if len(values) > 0:
                bin_medians.append(np.median(values))
                p15.append(np.percentile(values, 16))
                p85.append(np.percentile(values, 86))
            else:
                bin_medians.append(np.nan)
                p15.append(np.nan)
                p85.append(np.nan)
        bins_medians = np.array(bin_medians)
        p15 = np.array(p15)
        p85 = np.array(p85)
        lower_err = bin_medians - p15
        upper_err = p85 - bin_medians
        yerr = [lower_err, upper_err]

        #Now we're graphing
        fig, ax1 = plt.subplots(figsize=(7, 4))
        # LEFT AXIS: fraction histogram
        ax1.hist(below_50, bins=bins, histtype='step', color='black')
        ax1.set_xlabel("Response")
        ax1.set_ylabel("Counts", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # RIGHT AXIS: ecal_ratio vs fraction (binned)
        ax2 = ax1.twinx()
        ax2.errorbar(
            bin_centers,
            bin_medians,
            yerr=yerr,
            fmt='o',
            capsize=3,
            color='red'
        )
        ax2.set_ylabel("Ecal_ratio (median ± 15–85%)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.title(f"Response + binned Ecal_Ratio for {pid} {e} Gev {a} Degrees Ratio")
        plt.tight_layout()
        plt.savefig(f"below_ratio_{pid}_{e}_{a}_ratio.pdf")
        plt.legend()
        plt.tight_layout()
        plt.close()

        #####
        bins = np.linspace(0.6, 1.4, 30)
        digitized = np.digitize(above_50, bins) #assigns each thing a bin index 
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_medians = []
        p15 = []
        p85 = []
        for i in range(1, len(bins)): #digitize returns an index starting at 1 
            values = ecal_high[digitized == i]
            if len(values) > 0:
                bin_medians.append(np.median(values))
                p15.append(np.percentile(values, 16))
                p85.append(np.percentile(values, 86))
            else:
                bin_medians.append(np.nan)
                p15.append(np.nan)
                p85.append(np.nan)
        bins_medians = np.array(bin_medians)
        p15 = np.array(p15)
        p85 = np.array(p85)
        lower_err = bin_medians - p15
        upper_err = p85 - bin_medians
        yerr = [lower_err, upper_err]

        #Now we're graphing
        fig, ax1 = plt.subplots(figsize=(7, 4))
        # LEFT AXIS: fraction histogram
        ax1.hist(above_50, bins=bins, histtype='step', color='black')
        ax1.set_xlabel("Response")
        ax1.set_ylabel("Counts", color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # RIGHT AXIS: ecal_ratio vs fraction (binned)
        ax2 = ax1.twinx()
        ax2.errorbar(
            bin_centers,
            bin_medians,
            yerr=yerr,
            fmt='o',
            capsize=3,
            color='red'
        )
        ax2.set_ylabel("Ecal_ratio (median ± 15–85%)", color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.title(f"Response + binned Ecal_Ratio for {pid} {e} Gev {a} Degrees Ratio")
        plt.tight_layout()
        plt.savefig(f"above_ratio_{pid}_{e}_{a}_ratio.pdf")
        plt.legend()
        plt.tight_layout()
        plt.close()


#/a_pdg_211_pt_50_theta_135-135_trial5/reco_pdg_pdg_pt_pt_theta_theta_nobib.root

'''
