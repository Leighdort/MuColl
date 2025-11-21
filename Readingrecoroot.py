import uproot
import matplotlib.pyplot as plt
import os

file = uproot.open("/home/rldohert/tutorial2024/output_recodif.root")
print(file.keys())
print(file.classnames())