import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

# System IDs
ECAL_BARREL = 679272617
HCAL_BARREL = 1573202488
ECAL_ENDCAP = 3383333369
HCAL_ENDCAP = 2381985645

real_systems = [
    "EcalBarrelCollectionRec",
    "HcalBarrelCollectionRec",
    "EcalEndcapCollectionRec",
    "HcalEndcapCollectionRec"
]

particle = [211]
angles = [15, 85, 140]
ratio_211 = []
ratio_11 = []
high_211 = []
low_211 = []
high_11 = []
low_11 = []
for pid in particle:
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")

        file = uproot.open(
            f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial10/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root"
        )
        events = file["events"]

        pandora_clusters = events["PandoraClusters"]
        pandora_clusters_hits = events["_PandoraClusters_hits"]
        mcparticles = events["MCParticles"]

        # Load arrays once
        status_mc = mcparticles["MCParticles.generatorStatus"].array()

        mc_momx = mcparticles["MCParticles.momentum.x"].array()
        mc_momy = mcparticles["MCParticles.momentum.y"].array()
        mc_momz = mcparticles["MCParticles.momentum.z"].array()
        mc_mass = mcparticles["MCParticles.mass"].array()

        cluster_energy = pandora_clusters["PandoraClusters.energy"].array()

        hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
        hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

        collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
        hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()

        # Preload energy maps
        energy_map = {}
        for name in real_systems:
            prefix = f"{name}/{name}"
            energy_map[name] = events[f"{prefix}.energy"].array()

        fraction = []

        for i in range(events.num_entries):

            if len(cluster_energy[i]) != 1:
                continue

            mask = (status_mc[i] == 1)
            if not np.any(mask):
                continue

            momx = mc_momx[i][mask][0]
            momy = mc_momy[i][mask][0]
            momz = mc_momz[i][mask][0]
            mcmass = mc_mass[i][mask][0]

            mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
            mc_energy = np.sqrt(mcmass**2 + mc_momentum**2)

            hits_begin_arr = hits_begin_all[i]
            hits_end_arr   = hits_end_all[i]
            collection_ID  = collectionID_all[i]
            hit_index      = hit_index_all[i]

            EOE = 0.0

            for j in range(len(hits_begin_arr)):
                lo = hits_begin_arr[j]
                hi = hits_end_arr[j]

                sysIDs = collection_ID[lo:hi]
                idxs   = hit_index[lo:hi]

                # 🔥 vectorized masks
                mask_eb = (sysIDs == ECAL_BARREL)
                mask_hb = (sysIDs == HCAL_BARREL)
                mask_ee = (sysIDs == ECAL_ENDCAP)
                mask_he = (sysIDs == HCAL_ENDCAP)

                # 🔥 vectorized energy sums
                if np.any(mask_eb):
                    EOE += np.sum(energy_map["EcalBarrelCollectionRec"][i][idxs[mask_eb]])
                if np.any(mask_hb):
                    EOE += np.sum(energy_map["HcalBarrelCollectionRec"][i][idxs[mask_hb]])
                if np.any(mask_ee):
                    EOE += np.sum(energy_map["EcalEndcapCollectionRec"][i][idxs[mask_ee]])
                if np.any(mask_he):
                    EOE += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])

            fraction.append(EOE / mc_energy)

        fraction = np.array(fraction)
        '''
        # 📊 Histogram with log-scale y-axis
        plt.hist(fraction, bins=50, edgecolor='black')
        plt.yscale('log')   # ✅ log scale

        plt.xlabel("EOE / MC Energy")
        plt.ylabel("Count (log scale)")
        plt.title(f"{pid} at {a} degrees")

        plt.tight_layout()
        plt.savefig(f"fraction_{pid}_{a}.pdf")
        plt.close()
        '''
        median = np.median(fraction)
        ratio_211.append(median)
        q16, q84 = np.percentile(fraction, [16, 84])
        low_211.append(median-q16)
        high_211.append(q84-median)
particle = [11]
angles = [15, 85, 140]
for pid in particle:
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")

        file = uproot.open(
            f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial6/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root"
        )
        events = file["events"]

        pandora_clusters = events["PandoraClusters"]
        pandora_clusters_hits = events["_PandoraClusters_hits"]
        mcparticles = events["MCParticles"]

        # Load arrays once
        status_mc = mcparticles["MCParticles.generatorStatus"].array()

        mc_momx = mcparticles["MCParticles.momentum.x"].array()
        mc_momy = mcparticles["MCParticles.momentum.y"].array()
        mc_momz = mcparticles["MCParticles.momentum.z"].array()
        mc_mass = mcparticles["MCParticles.mass"].array()

        cluster_energy = pandora_clusters["PandoraClusters.energy"].array()

        hits_begin_all = pandora_clusters["PandoraClusters.hits_begin"].array()
        hits_end_all   = pandora_clusters["PandoraClusters.hits_end"].array()

        collectionID_all = pandora_clusters_hits["_PandoraClusters_hits.collectionID"].array()
        hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()

        # Preload energy maps
        energy_map = {}
        for name in real_systems:
            prefix = f"{name}/{name}"
            energy_map[name] = events[f"{prefix}.energy"].array()

        fraction = []

        for i in range(events.num_entries):

            if len(cluster_energy[i]) != 1:
                continue

            mask = (status_mc[i] == 1)
            if not np.any(mask):
                continue

            momx = mc_momx[i][mask][0]
            momy = mc_momy[i][mask][0]
            momz = mc_momz[i][mask][0]
            mcmass = mc_mass[i][mask][0]

            mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
            mc_energy = np.sqrt(mcmass**2 + mc_momentum**2)

            hits_begin_arr = hits_begin_all[i]
            hits_end_arr   = hits_end_all[i]
            collection_ID  = collectionID_all[i]
            hit_index      = hit_index_all[i]

            EOE = 0.0

            for j in range(len(hits_begin_arr)):
                lo = hits_begin_arr[j]
                hi = hits_end_arr[j]

                sysIDs = collection_ID[lo:hi]
                idxs   = hit_index[lo:hi]

                # 🔥 vectorized masks
                mask_eb = (sysIDs == ECAL_BARREL)
                mask_hb = (sysIDs == HCAL_BARREL)
                mask_ee = (sysIDs == ECAL_ENDCAP)
                mask_he = (sysIDs == HCAL_ENDCAP)

                # 🔥 vectorized energy sums
                if np.any(mask_eb):
                    EOE += np.sum(energy_map["EcalBarrelCollectionRec"][i][idxs[mask_eb]])
                if np.any(mask_hb):
                    EOE += np.sum(energy_map["HcalBarrelCollectionRec"][i][idxs[mask_hb]])
                if np.any(mask_ee):
                    EOE += np.sum(energy_map["EcalEndcapCollectionRec"][i][idxs[mask_ee]])
                if np.any(mask_he):
                    EOE += np.sum(energy_map["HcalEndcapCollectionRec"][i][idxs[mask_he]])

            fraction.append(EOE / mc_energy)

        fraction = np.array(fraction)
        '''
        # 📊 Histogram with log-scale y-axis
        plt.hist(fraction, bins=50, edgecolor='black')
        plt.yscale('log')   # ✅ log scale

        plt.xlabel("EOE / MC Energy")
        plt.ylabel("Count (log scale)")
        plt.title(f"{pid} at {a} degrees")

        plt.tight_layout()
        plt.savefig(f"fraction_{pid}_{a}.pdf")
        plt.close()
        '''
        median = np.median(fraction)
        ratio_11.append(median)
        q16, q84 = np.percentile(fraction, [16, 84])
        low_11.append(median-q16)
        high_11.append(q84-median)

    #Summary graph, number of bib clusters given 1 normal cluster
plt.errorbar(angles, ratio_211, yerr=[low_211, high_211], fmt='s', alpha= 0.6, capsize=4, label=f"Pions")
plt.errorbar(angles, ratio_11, yerr=[low_11, high_11], fmt='s', alpha= 0.6, capsize=4, label=f"Electrons")
plt.xlabel("Particle gun angle (degrees)")
plt.ylabel("Median Summed Hit Energy / Cluster Energy ")
plt.title(f" Total hit Energy / Cluster Energy for 50 GeV ")
#plt.title(f" Original 50 GeV Median Response 1 Nobib Cluster Pid: {pid} ")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"summary_hitclus.pdf")
plt.close()





'''
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
angles = [15, 85, 140]
#angles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
yes = True
#This is exluding bounds!
for pid in particle:
    mc_median= []
    mc_low = []
    mc_high = []
    for a in angles:
        print(f"\n=== Energy {a} Degrees ===")
        fraction = []
        energy_clus = []
        energy_hits = []
        num_events_1_clus = 0
        passes = 0
        file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/a_pdg_{pid}_pt_50_theta_{a}-{a}_trial6/job_0/reco_output_p50_{pid}_nobib0.edm4hep.root")
        events = file["events"]
        pandora_clusters = events["PandoraClusters"]
        #I need the theta phi of mc and of clusters s
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
        hit_index_all    = pandora_clusters_hits["_PandoraClusters_hits.index"].array()
        ecal_e_ratio = []
        hcal_e_ratio = []
        ecal_b_ratio = []
        hcal_b_ratio = []
        ecal_ratio = []
        energy_map = {}
        for name in real_systems:
            prefix = f"{name}/{name}"
            energy_map[name] = events[f"{prefix}.energy"].array()

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
                clus_clus_energy = cluster_energy[i]
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
                cosang = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                cosang = np.clip(cosang, -1.0, 1.0)
                angular_distance = np.arccos(cosang)
                other_angle = (np.sin(mc_theta)*np.sin(c_theta)*np.cos(mc_phi - c_phi)+ np.cos(mc_theta)*np.cos(c_theta))
                other_dist = np.sqrt(mc_r**2 + c_r**2 - 2*mc_r*c_r*other_angle)
                #if angular_distance <= bound:
                if yes == True: 
                    #print("hi")
                    passes += 1
                    mc_momentum = np.sqrt(momx**2 + momy**2 + momz**2)
                    mc_energy = np.sqrt(mcmass*mcmass + mc_momentum*mc_momentum)
                    #To find fraction we're going to have to recalibrate everything by hand "yay :)"
                    hits_begin_arr = hits_begin_all[i]
                    hits_end_arr = hits_end_all[i]
                    hit_index = hit_index_all[i]
                    collection_ID = collectionID_all[i]
                    EOE = 0.0 #energy of event
                    for j in range(len(hits_begin_arr)):
                        lo = hits_begin_arr[j]
                        hi = hits_end_arr[j]
                        sysIDs = collection_ID[lo:hi]
                        idxs = hit_index[lo:hi]
                        #print(idxs)
                        ##WE NEED TO FIX HIT INDEX HERE THATS WHAT IT SHOULD BE
                        #[name][i][idx]!!!!!
                        
                        for p, code in enumerate(sysIDs):
                            name = system2name.get(code, "Skip")
                            if name == "Skip":
                                continue
                            #energy = energy_map[name][i][j]
                            #LET"S TEST BUT PROBBALY FAIL
                            idx = idxs[p]
                            energy = energy_map[name][i][idx]
                            if code == 679272617:
                                #print ("hi")
                                #EOE += energy * 0.9
                                EOE += energy
                                ecal_barrel +=1
                            elif code == 1573202488:
                                hcal_barrel +=1
                                #EOE += energy * 1.12
                                EOE += energy
                                #print ("hii")
                            elif code == 3383333369:
                                ecal_endcap +=1
                                #EOE += energy * 0.95
                                EOE += energy
                                #print ("hiii")
                            elif code == 2381985645:
                                #EOE += energy * 1.02
                                EOE += energy
                                hcal_endcap +=1
                                #print ("hiiii")
                    #here we find fraction
                    #We will have to do appendation like this:
                    energy_hits.append(EOE)
                    energy_clus.append(clus_clus_energy)
                    fraction_mc = EOE / mc_energy[0]
                    fraction.append(fraction_mc)
                    total = ecal_barrel + hcal_barrel + ecal_endcap + hcal_endcap
                    ecal_total = ecal_barrel + ecal_endcap
                    if total != 0:
                        ecal_ratio.append(ecal_total / total)
                    else:
                        ecal_ratio.append(0)
        fraction = np.array(fraction)
        ecal_ratio = np.array(ecal_ratio)
        #Then want to do an ecal_endcap vs ecal_barrel check
        #Let's simply cut on above or 50 for one graph
        #50 for another
        below_50 = []
        above_50 = []
        ecal_low = []
        ecal_high = []
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
        plt.title(f"Below 0.50 ecal Ratio {a} {pid} Nobib Energy 50 GeV")
        plt.tight_layout()
        plt.savefig(f"base10_below_{a}_fraction_nobib_{pid}50GeV_trial6.pdf")
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
        plt.title(f"Above 0.50 ecal Ratio {a} {pid} Nobib Energy 50 GeV")
        plt.tight_layout()
        plt.savefig(f"base10_above_{a}_fraction_nobib_{pid}50GeV_trial6.pdf")
        plt.close()

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
        plt.title(f"Response Comparison ({pid}, {a}°, 50 GeV)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"base10_combined_{a}_fraction_{pid}_50GeV_trial6.pdf")
        plt.close()
        
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
        plt.title(f"Response + binned Ecal_Ratio for {pid} {e} Gev {a} Degrees")
        plt.tight_layout()
        plt.savefig(f"base10_below_ratio_{pid}_{e}_{a}_trial6.pdf")
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
        plt.title(f"Response + binned Ecal_Ratio for {pid} {e} Gev {a} Degrees")
        plt.tight_layout()
        plt.savefig(f"base10_above_ratio_{pid}_{e}_{a}_trial6.pdf")
        plt.legend()
        plt.tight_layout()
        plt.close()
        

        #Now graph energy_hits vs energy_clus
        energy_hits = np.array(energy_hits)
        energy_clus = np.array(energy_clus)
        energy_ratio = energy_hits / energy_clus

        # Define bins
        bins = np.linspace(0, 2, 40)  # adjust range if needed
        # Get counts
        counts, bin_edges = np.histogram(energy_ratio, bins=bins)
        # Bin centers for plotting
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Plot
        plt.figure(figsize=(7,5))
        plt.bar(bin_centers, counts, width=(bin_edges[1] - bin_edges[0]), edgecolor='black')
        plt.xlabel("Energy Ratio (Hits / Cluster)")
        plt.ylabel("Counts")
        plt.title(f"Energy Ratio Distribution {pid}, {e}, {a}")
        plt.tight_layout()
        plt.savefig(f"energy_ratio_hist_{pid}_{e}_{a}.pdf")
        plt.close()
'''




#/a_pdg_211_pt_50_theta_135-135_trial5/reco_pdg_pdg_pt_pt_theta_theta_nobib.root





 


