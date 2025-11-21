from pyLCIO import IOIMPL

n_events_to_keep = 100
energies = [10, 50, 100, 150, 200]

'''for E in energies:
    input_file = f"outputelectron_sim{E}.slcio"
    output_file = f"outputelectron_sim{E}_100.slcio"

    print(f"Processing {input_file} → {output_file}")

    reader = IOIMPL.LCFactory.getInstance().createLCReader()
    writer = IOIMPL.LCFactory.getInstance().createLCWriter()

    reader.open(input_file)
    writer.open(output_file)  # no enum needed

    for i, event in enumerate(reader):
        if i >= n_events_to_keep:
            break
        writer.writeEvent(event)
        if (i + 1) % 10 == 0:
            print(f"  Copied {i+1} events...")

    reader.close()
    writer.close()

    print(f"✅ Done! Wrote first {n_events_to_keep} events to {output_file}\n")'''

import subprocess
import os

n_events = 100
signal_file = "outputelectron_sim10_100.slcio"
muplus = "/home/rldohert/tutorial2024/BIB/MUPLUS"
muminus = "/home/rldohert/tutorial2024/BIB/MUMINUS"

for i in range(n_events):
    # Run k4run on one event at a time
    cmd = [
        "k4run",
        "mucoll-benchmarks/digitisation/k4run/digi_steer.py",
        "--LcioEvent.Files", signal_file,
        "--doOverlayFull",
        "--OverlayFullPathToMuPlus", muplus,
        "--OverlayFullPathToMuMinus", muminus,
        "--OverlayFullNumberBackground", "19",
        "-n", "1",         # process 1 event at a time
    ]

    subprocess.run(cmd, check=True)

    # Rename the output file
    os.rename("output_digi.slcio", f"outputelectronsigi10_{i}_with10bib.slcio")

# Merge all temporary overlay files
merge_cmd = "lcio_merge outputelectronsigi10_*_with10bib.slcio merged_output.slcio"
subprocess.run(merge_cmd, shell=True, check=True)
