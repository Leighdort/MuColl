#This is going to plot given the summery files

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
particle = [11, 211]
#angles = [15, 85, 140]
angles = [15, 85]
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
        file = uproot.open(f"/users/rldohert/work/mucoll/mucoll-slurm/files/matched_clusters_pid{pid}_E50_theta{a}_software_on_nobib.root")
        tree = file["MatchedClusters"]
        arrays = tree.arrays(library="np")
        matched_energy = arrays["matched_energy"]
        mc_energy = arrays["mc_energy"]
        ecal_barrel = arrays["ecal_barrel"]
        hcal_barrel = arrays["hcal_barrel"]
        ecal_endcap = arrays["ecal_endcap"]
        hcal_endcap = arrays["hcal_endcap"]
        print(ecal_barrel)
        print(hcal_barrel)
        print(ecal_endcap)
        print(hcal_endcap)
        mc_energy = np.asarray(mc_energy).reshape(-1)
        matched_energy = np.asarray(matched_energy).reshape(-1)
        ecal_barrel = np.asarray(ecal_barrel).reshape(-1)
        hcal_barrel = np.asarray(hcal_barrel).reshape(-1)
        ecal_endcap = np.asarray(ecal_endcap).reshape(-1)
        hcal_endcap = np.asarray(hcal_endcap).reshape(-1)
        print(ecal_barrel)
        print(hcal_barrel)
        print(ecal_endcap)
        print(hcal_endcap)
        fraction = matched_energy / mc_energy
        ecal_ratio = ecal_barrel + ecal_endcap
        hcal_ratio = hcal_barrel + hcal_endcap
        total = ecal_ratio + hcal_ratio
        ecal_ratio = ecal_ratio / total
        hcal_ratio = hcal_ratio / total
        print(len(fraction))
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
        bins = np.linspace(0, 2, 60)
        below_50 = np.array(below_50)
        above_50 = np.array(above_50)
        ecal_low = np.array(ecal_low)
        ecal_high = np.array(ecal_high)
        median_below = np.median(below_50)
        median_above = np.median(above_50)
        median = np.median(fraction)
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
        '''
        #This one we're trying shift the calibrated This is flaming trash 
        for i in range(len(fraction)):
            baseline = fraction[i]
            correction_1 = ecal_ratio[i] * 0.13
            correction_2 =  hcal_ratio[i] * .10
            new_fraction = baseline - correction_1 + correction_2
            fraction[i] = new_fraction
        '''
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
        #Then want to do an ecal_endcap vs ecal_barrel check
        #Let's simply cut on above or 50 for one graph
        #50 for another
        

        #Can we plot these on the same graph with different opacities adn colors ontop of eachother
        #One labeled below 50
        #One labeled above 50
        bins = np.linspace(0, 2, 60)
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

        plt.axvline(
            median,
            linestyle='--',
            linewidth=2,
            label=f'Median = {median:.3f}'
        )

        plt.xlabel("Response")
        plt.ylabel("Count")
        plt.xlim(0, 2)
        plt.title(f"Response ({pid}, {a}°, 50 GeV) Hit Baseline NoBib Software On")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"combined_{a}_fraction_{pid}_hit_baseline_nobib_softon.pdf")
        plt.close()
        #Let's just plot fraction now w/ energy

        #Now just a normal graph: 
        q16, q84 = np.percentile(fraction, [16, 84])
        median = np.median(fraction)
        #But we also want to graph displays
        #bins=30
        bins = np.linspace(0, 2, 60)  # 60 bins → need 61 edges
        plt.xlim(0, 2)
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
        plt.title(f"Response {a} deg PID: {pid} 50 GeV Hit Baseline NoBib Software On")
        plt.tight_layout()
        plt.savefig(f"ratio_{a}_gaussian_{pid}50GeV_hit_baseline_nobib_softon.pdf")
        plt.close()

