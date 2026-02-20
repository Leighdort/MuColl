#Making graphs
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

energy = [2, 10, 50]
nobib_median=[0.11045,0.13186,0.14271]
nobib_low=[0.03804,0.04325,0.04129]
nobib_high=[0.04668,0.04052,0.03285]
bib_median=[0.14273,0.14272,0.14269]
bib_high=[0.02920,0.02903,0.02915]
bib_low=[0.03577,0.03009,0.03003]
plt.errorbar(energy, nobib_median, yerr=[nobib_low, nobib_high], fmt='s', alpha=0.6, capsize=4, label="Events without Bib")
plt.errorbar(energy, bib_median, yerr=[bib_low, bib_high], fmt='s', alpha=0.6, capsize=4, label="Events with Bib")
plt.xlabel("Beam Energy")
plt.ylabel("Event Width")
plt.title("Event width for Pions with Bib and without Bib")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("summary_widthbib.pdf")
plt.close

