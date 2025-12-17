#Printing files
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

'''
energies = [1, 2, 5, 10, 50, 100, 150, 200]
electron_info_miss = [0,0,0,0,0,0,0.0005786353,0.00069044356]
electron_std_miss = [0, 0, 0, 0, 0, 0,0.00029624294,0]
electron_info_ratio = [0,0,0,0,0,0,1.91455163e-06, 1.0076259e-06]
electron_std_ratio = [0,0,0,0,0,0,1.4069961e-06,0]
pion_info_miss = [0, 0.0009443901, 0.001410144, 0.0013133102, 0.0024419392, 0.0029912659, 0.0033331513, 0.0035017321]
pion_std_miss = [0, 0.0, 0.0012682051, 0.0015092281, 0.0060342625, 0.006743244, 0.0065694037, 0.006882778]
pion_info_ratio = [0, 0.00013167952, 0.0000674254, 0.00007686458, 0.000018252635, 0.00001738953, 0.0000070982196, 0.00000632725]
pion_std_ratio = [0, 0.0, 0.000055184868, 0.0002103445, 0.00007696671, 0.00023475387, 0.000027816512, 0.00004324816]
types = 2032
electrons = [0,0,0,0,0,0,9,7]
pions = [0,3,82,393,6716,18988,33050,45838]

#We should also add particle type
plt.errorbar(energies, electron_info_miss, yerr=electron_std_miss, fmt='o', capsize=4, label="Electrons")
plt.errorbar(energies, pion_info_miss, yerr=pion_std_miss, fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy (GeV)")
plt.ylabel("Missed Energy")
plt.title("Missed Energy vs Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("missed_energy_vs_energy.pdf")
plt.close()

plt.errorbar(energies, electron_info_ratio, yerr=electron_std_ratio, fmt='o', capsize=4, label="Electrons")
plt.errorbar(energies, pion_info_ratio, yerr=pion_std_ratio, fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy (GeV)")
plt.ylabel("Missed / Total Ratio")
plt.title("Missed/Total Energy Ratio vs Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("missed_ratio_vs_energy.pdf")
plt.close()

plt.plot(energies, electrons, 'o-', label="Electrons")
plt.plot(energies, pions, 's-', label="Pions")
plt.xlabel("Beam Energy (GeV)")
plt.ylabel(f"Particle Counts (Type {types})")
plt.title("Particle Counts vs Beam Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("particle_counts_vs_energy.pdf")
plt.close()
'''
'''
energies = [1, 2, 5, 10, 50, 100, 150, 200]
electron_info = [0.03281,0.03096,0.02966,0.02964,0.03233,0.03228,0.03230,0.03247]
electron_std = [0.01034,0.00582,0.00571,0.00635,0.00843,0.00796,0.00703,0.00703]
pion_info = [0.11279,0.11553,0.12668,0.13100,0.13880,0.14297,0.14452,0.14476]
pion_std = [0.04863,0.04516,0.04445,0.04274,0.03766,0.03526,0.03408,0.03431]

plt.errorbar(energies, electron_info, yerr=electron_std, fmt='o', capsize=4, label="Electrons")
plt.errorbar(energies, pion_info, yerr=pion_std, fmt='s', capsize=4, label="Pions")
plt.xlabel("Beam Energy")
plt.ylabel("Width")
plt.title("Average width per energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_width.pdf")
plt.close()
'''
energies = [1, 2, 5, 10, 50, 100, 150, 200]
# Mean ECal energies
elec_mean_ecal = [0.6207181811332703, 0.8626291751861572, 1.4255902767181396, 2.1203510761260986,
             4.791934967041016, 6.3585333824157715, 7.190113067626953, 7.708284378051758]
# Mean HCal energies
elec_mean_hcal = [9.811707496643066, 9.339003562927246, 9.558516502380371, 9.4028959274292,
             9.043529510498047, 8.988040924072266, 8.966110229492188, 8.961085319519043]
# Stddev ECal energies
elec_std_ecal = [0.8839275240898132, 1.144112467765808, 1.6336638927459717, 2.149731159210205,
            2.7776906490325928, 2.4482004642486572, 2.093783140182495, 1.7687077522277832]
# Stddev HCal energies
elec_std_hcal = [6.8369059562683105, 1.7655836343765259, 4.306597709655762, 3.0265491008758545,
            0.5549065470695496, 0.875796377658844, 0.017643479630351067, 0.010108085349202156]

#Ok now the same for pions;

# Mean ECal energies
pion_mean_ecal = [4.607046127319336, 5.0492024421691895, 5.725510120391846, 6.150203704833984,
             7.2506513595581055, 7.739747047424316, 7.96605920791626, 8.214613914489746]
# Mean HCal energies
pion_mean_hcal = [10.17436695098877, 9.502426147460938, 9.039555549621582, 8.983073234558105,
             8.959487915039062, 8.959606170654297, 8.958731651306152, 8.958168029785156]
# Stddev ECal energies
pion_std_ecal = [3.211388111114502, 3.3588998317718506, 3.425992727279663, 3.496840238571167,
            3.426414966583252, 3.2594592571258545, 3.15775990486145, 2.9915571212768555]
# Stddev HCal energies
pion_std_hcal = [4.932833194732666, 3.1433820724487305, 1.1571003198623657, 0.43229490518569946,
            0.022027665749192238, 0.013457683846354485, 0.013496596366167068, 0.01357179507613182]

plt.errorbar(energies, elec_mean_ecal, yerr=elec_std_ecal, fmt='o', capsize=4, label="Electrons Average Ecal End")
plt.errorbar(energies, elec_mean_hcal, yerr=elec_std_hcal, fmt='s', capsize=4, label="Electrons Average Hcal Start")
plt.errorbar(energies, pion_mean_ecal, yerr=pion_std_ecal, fmt='o', capsize=4, label="Pions Average Ecal End")
plt.errorbar(energies, pion_mean_hcal, yerr=pion_std_hcal, fmt='s', capsize=4, label="Pions Average Hcal Start")
plt.xlabel("Beam Energy")
plt.ylabel("Time")
plt.title("Average Time for Exit for Entry for a Cluster")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_time_cluster.pdf")
plt.close()
