#Check mc
#This was not written by me!, as this is time consuming and annoying
#We will print out the median, low, high for x, y, z, energy 
#We will check the positions for the bib and no bib file and make sure they are the same

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
energies = [2, 10, 50]
choices = [2, 10, 50]
x_m = []
x_l = []
x_h = []
y_m = []
y_l = []
y_h = []
z_m = []
z_l = []
z_h = []
m_m = []
m_h = []
m_l = []

for num in energies:
    print(f"\n=== Energy {num} GeV ===")
    file = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_nobib.root")
    file_bib = uproot.open(f"/users/rldohert/data/mucoll/rldohert/pdg_211_pt_{num}_theta_15-15_bib2/reco_pdg_211_pt_{num}_theta_15-15_bib.root")
    bib_events = file_bib["events"]
    events = file["events"]
    mcparticles = events["MCParticles"]
    bibmcparticles = bib_events["MCParticles"]
    status_mc = mcparticles["MCParticles.generatorStatus"].array()
    bibstatus_mc = bibmcparticles["MCParticles.generatorStatus"].array()
    mc_x = mcparticles["MCParticles.endpoint.x"].array()
    mc_y = mcparticles["MCParticles.endpoint.y"].array()
    mc_z = mcparticles["MCParticles.endpoint.z"].array()
    mc_mass = mcparticles["MCParticles.mass"].array()
    bmc_x = bibmcparticles["MCParticles.endpoint.x"].array()
    bmc_y = bibmcparticles["MCParticles.endpoint.y"].array()
    bmc_z = bibmcparticles["MCParticles.endpoint.z"].array()
    bmc_mass = bibmcparticles["MCParticles.mass"].array()

    diff_x = []
    diff_y = []
    diff_z = []
    diff_m = []

    for i in range(events.num_entries):
        mask = (status_mc[i] == 1)
        bmask = (bibstatus_mc[i] == 1)
        mx = mc_x[i][mask]
        my = mc_y[i][mask]
        mz = mc_z[i][mask]
        mm = mc_mass[i][mask]
        bx = bmc_x[i][bmask]
        by = bmc_y[i][bmask]
        bz = bmc_z[i][bmask]
        bm = bmc_mass[i][bmask]
        # Only compare if same number of mc particles
        if len(mx) != len(bx):
            continue
        for j in range(len(mx)):
            diff_x.append(mx[j] - bx[j])
            diff_y.append(my[j] - by[j])
            diff_z.append(mz[j] - bz[j])
            diff_m.append(mm[j] - bm[j])

    for label, diff, med_list, low_list, high_list in [
        ("x", diff_x, x_m, x_l, x_h),
        ("y", diff_y, y_m, y_l, y_h),
        ("z", diff_z, z_m, z_l, z_h),
        ("mass", diff_m, m_m, m_l, m_h),
    ]:
        arr = np.array(diff)
        median = np.median(arr)
        q16, q84 = np.percentile(arr, [16, 84])
        med_list.append(median)
        low_list.append(median - q16)
        high_list.append(q84 - median)
        print(f"  {label}: median={median:.4f}, low={median-q16:.4f}, high={q84-median:.4f}")

        plt.hist(arr, bins=30, edgecolor='black')
        plt.axvline(median, color='red', linestyle='--', linewidth=2, label=f"Median = {median:.4f}")
        plt.xlabel(f"Nobib - Bib MC endpoint {label}")
        plt.ylabel("Count")
        plt.title(f"MC Endpoint {label} Difference Pion {num} GeV")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"mc_check_{label}_{num}GeV.pdf")
        plt.close()

# Summary plots
for label, med_list, low_list, high_list in [
    ("x", x_m, x_l, x_h),
    ("y", y_m, y_l, y_h),
    ("z", z_m, z_l, z_h),
    ("mass", m_m, m_l, m_h),
]:
    plt.errorbar(choices, med_list, yerr=[low_list, high_list], fmt='s', alpha=0.6, capsize=4, label="Pion particle Gun")
    plt.xlabel("Beam Energy (GeV)")
    plt.ylabel(f"Median Nobib - Bib MC {label} Difference")
    plt.title(f"MC Endpoint {label} Difference vs Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"summary_mc_check_{label}.pdf")
    plt.close()