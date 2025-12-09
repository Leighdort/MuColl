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