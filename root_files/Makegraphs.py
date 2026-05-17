#Making graphs
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

#Submit_maker.py Submits this
energy = [2, 10, 50]
nobib_median=[0.11486312747001648,0.13754183053970337,0.15152689814567566]
nobib_low=[0.036539177000522616,0.039085880517959595,0.03452596873044968]
nobib_high=[0.04583398282527923,0.03782742857933044,0.028579987883567803]
bib_median=[0.15083307027816772,0.14449664950370789,0.15203602612018585]
bib_high=[0.04183481037616729,0.03787957549095153,0.029584470987319922]
bib_low=[0.0350660815834999,0.03689909309148788,0.03381349831819534]
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

