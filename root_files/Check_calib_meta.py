#Checking the metadata and how it is stored 


# Checking the metadata and how it is stored

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uproot

# Open ROOT file
file = uproot.open(
    "/users/rldohert/data/mucoll/rldohert/calib_test_work/job_0/reco_output_pt10_200_211_nobib0.edm4hep.root"
)

# Print all tree names / objects stored in the file
print("\n========== TREE NAMES ==========")
print(file.keys())

# Access events tree
events = file["events"]

# Print all possible branches under the events tree
print("\n========== EVENTS BRANCHES ==========")
for branch in events.keys():
    print(branch)

# Print number of events
print("\n========== EVENT INFORMATION ==========")
print("Number of events:", events.num_entries)

# Print detailed tree structure, including branch types
print("\n========== TREE STRUCTURE ==========")
events.show()
