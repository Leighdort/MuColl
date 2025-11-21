# Getting Truth Particles
import argparse
import math
import pyLCIO
import csv
from pyLCIO import UTIL, EVENT
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

X, Y, Z = 0, 1, 2

def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, type=str, help="Input LCIO file")
    parser.add_argument("-o", required=True, type=str, help="Output CSV base name")
    parser.add_argument("-n", required=False, type=int, help="Number of events to process")
    return parser.parse_args()

def get_theta(px, py, pz):
    pt = math.sqrt(px**2 + py**2)
    return math.atan2(pt, pz)

def get_phi(px, py):
    return math.atan2(py, px)

def get_collection(event, name):
    names = event.getCollectionNames()
    if name in names:
        return event.getCollection(name)
    return []

def main():
    ops = options()
    print(f"Reading file {ops.i}")
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(ops.i)

    all_mcps = []

    for i_event, event in enumerate(reader):
        if ops.n is not None and i_event >= ops.n:
            break

        mc_particles = get_collection(event, "MCParticle")
        event_mcps = []

        for i_mcp, mcp in enumerate(mc_particles):
            pdg = mcp.getPDG()                     # PDG code (13 = mu-, -13 = mu+)
            energy = mcp.getEnergy()               # total energy [GeV]
            mass = mcp.getMass()                   # particle mass [GeV]
            charge = mcp.getCharge()               # charge [e]
            momentum = mcp.getMomentum()           # (px, py, pz)
            px, py, pz = momentum[X], momentum[Y], momentum[Z]
            vertex = mcp.getVertex()               # production vertex (x, y, z)
            endpoint = mcp.getEndpoint()           # decay/exit point (x, y, z)
            generatorStatus = mcp.getGeneratorStatus()  # 1 = stable final state
            time = mcp.getTime()                   # production time [ns]
            parents = mcp.getParents()             # list of parent MCParticles
            daughters = mcp.getDaughters()         # list of daughter MCParticles

            # Angles
            theta = get_theta(px, py, pz)
            phi = get_phi(px, py)

            mcp_info = {
                "event": i_event,
                "mcp_num": i_mcp,
                "pdg": pdg,
                "charge": charge,
                "energy": energy,
                "mass": mass,
                "momentum": (px, py, pz),
                "vertex": (vertex[X], vertex[Y], vertex[Z]),
                "endpoint": (endpoint[X], endpoint[Y], endpoint[Z]),
                "generatorStatus": generatorStatus,
                "time": time,
                "num_parents": len(parents),
                "num_daughters": len(daughters),
                "theta": theta,
                "phi": phi
            }
            event_mcps.append(mcp_info)

        all_mcps.extend(event_mcps)

    # --- Write MCParticles to CSV ---
    with open(ops.o.replace(".csv", "_mcparticles.csv"), "w", newline="") as f_mcps:
        mcps_fieldnames = [
            "event", "mcp_num", "pdg", "charge", "energy", "mass", "momentum",
            "vertex", "endpoint", "generatorStatus", "time",
            "num_parents", "num_daughters", "theta", "phi"
        ]
        writer = csv.DictWriter(f_mcps, fieldnames=mcps_fieldnames)
        writer.writeheader()
        for mcp in all_mcps:
            writer.writerow(mcp)

    print(f"Wrote {len(all_mcps)} MCParticles to {ops.o.replace('.csv', '_mcparticles.csv')}")

if __name__ == "__main__":
    main()
