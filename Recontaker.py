#I'm trying to get this to show all of the tracks
import argparse
import math
import pyLCIO
import csv
from pyLCIO import UTIL, EVENT
import pandas as pd
import numpy as np
import math
from collections import defaultdict
import matplotlib.pyplot as plt

X, Y, Z = 0, 1, 2
TRACK_COL = "SiTracks"
CLUSTER_COL = "PandoraClusters"
PFO_COL = "PandoraPFOs"
BFIELD = 5.0
FACTOR = 3e-4

CAL_COLLECTIONS = [
    "EcalBarrelCollectionDigi",
    "EcalEndcapCollectionDigi",
    "HcalBarrelCollectionDigi",
    "HcalEndcapCollectionDigi",
    "YokeBarrelCollection",
    "YokeEndcapCollection"
]


system2name = {
    20: "ecal barrel",
    29: "ecal endcap",
    10: "hcal barrel",
    11: "hcal barrel",
    13: "yoke barrel",
    14: "yoke endcap"
    #Problem some are 13 and 14
}
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", required=True, type=str, help="Input LCIO file")
    parser.add_argument("-o", required=True, type=str, help="Output LCIO file")
    parser.add_argument("-n", required=False, type=int, help="Number of events to process")
    return parser.parse_args()

def main():
    unknown_hits = []
    ops = options()
    print(f"Reading file {ops.i}")
    reader = pyLCIO.IOIMPL.LCFactory.getInstance().createLCReader()
    reader.open(ops.i)
    all_tracks = []
    all_clusters = []
    all_pfos = []
    all_hits = []
    for i_event, event in enumerate(reader):
        if ops.n is not None and i_event >= ops.n:
            break
        tracks = get_collection(event, TRACK_COL)
        event_tracks = []
        event_clusters = []
        event_pfos = []

        #Trying to understand and get the encoding string and stuff 
        #See where this should go 
        decoder = None
        for calo_col_name in CAL_COLLECTIONS: #Looping over names
            calo_hits_collection = get_collection(event, calo_col_name)
            if calo_hits_collection:
                encoding_str = calo_hits_collection.getParameters().getStringVal(EVENT.LCIO.CellIDEncoding)
                decoder = UTIL.BitField64(encoding_str)
                break

        clusters = get_collection(event, CLUSTER_COL)
        pfos = get_collection(event, PFO_COL)
        for i_track, track in enumerate(tracks):
            name = "track"
            omega, tan_lambda, phi = (
                track.getOmega(), #ok this is assigning stuff
                track.getTanLambda(),
                track.getPhi(),
            )
            theta = (math.pi / 2) - math.atan(tan_lambda)
            pt = BFIELD * FACTOR / abs(omega)
            pz = pt * tan_lambda
            p = math.sqrt(pt * pt + pz * pz)
            chi2, ndf = track.getChi2(), track.getNdf()
            track_info = {
                "event": i_event,
                "name": name,
                "track_num": i_track,
                "p": p,
                "pt": pt,
                "pz": pz,
                "theta": theta,
                "phi": phi,
                "chi2/ndf": f"{chi2}/{ndf}",
                }
            event_tracks.append(track_info)
        all_tracks.extend(event_tracks)
        #print(dir(clus)) gives all avaliable things





        for i_clus, clus in enumerate(clusters):
            name = "cluster"
            position = clus.getPosition() #Here is where you might get size 
            hits = clus.getCalorimeterHits()
            energy, theta, phi = clus.getEnergy(), clus.getITheta(), clus.getIPhi()
            for i_hit, hit in enumerate(hits):
                #if (i_hit == 1):
                #    print(dir(hit))
                pos = hit.getPosition()
                x, y, z = pos[X], pos[Y], pos[Z]
                cell_id = hit.getCellID0()
                decoder.setValue(cell_id)
                layer = decoder["layer"].value()
                location = system2name.get(decoder["system"].value(), "unknown")
                if layer == 0: #May cause a problem with an actual layer 0, but currently is functioning 
                    raw_cell_id = hit.getCellID0()
                    unknownhit = {
                        "event": i_event,
                        "hit_num": i_hit,
                        "energy": hit.getEnergy(),
                        "cluster_num": i_clus,
                        "cell_id": raw_cell_id,
                        "location": location,
                        "layer": layer,
                        "position": (position[X], position[Y], position[Z]),
                    }
                    unknown_hits.append(unknownhit)
                hit_info={
                    "event": i_event,
                    "cluster_num": i_clus,
                    "hit_num": i_hit,
                    "x": x,
                    "y": y,
                    "z": z,
                    "energy": hit.getEnergy(),
                    "time": hit.getTime(),
                    "cell_id": hit.getCellID0(),
                    "system": location,
                    "layer": layer
                }
                all_hits.append(hit_info)
            cluster_info = {
                "event": i_event,
                "name": name,
                "cluster_num": i_clus,
                "energy": energy,
                "theta": theta,
                "phi": phi,
                "position": (position[X], position[Y], position[Z]),
                "num_hits": len(hits),
            }
            event_clusters.append(cluster_info)
        all_clusters.extend(event_clusters)

        for i_pfo, pfo in enumerate(pfos):
            name = "PFO"
            momentum, energy = pfo.getMomentum(), pfo.getEnergy()
            pdg, charge = pfo.getType(), pfo.getCharge()
            px, py, pz = momentum[X], momentum[Y], momentum[Z]
            theta, phi = get_theta(px, py, pz), get_phi(px, py)
            pfo_info = {
                "event": i_event,
                "name": name,
                "pfo_num": i_pfo,
                "energy": energy,
                "momentum": (px, py, pz),
                "pdg": pdg,
                "charge": charge,
                "theta": theta,
                "phi": phi
            }
            event_pfos.append(pfo_info)
        all_pfos.extend(event_pfos)


    #Writing tracks 
    with open(ops.o.replace(".csv", "_tracks.csv"), "w", newline="") as f_tracks:
        tracks_fieldnames = [
            "event", "name", "track_num", "p",
            "pt", "pz", "theta", "phi",
            "chi2/ndf"
            ]
        writer = csv.DictWriter(f_tracks, fieldnames=tracks_fieldnames)
        writer.writeheader()
        for track in all_tracks:
            writer.writerow(track)
    
    #Writing clusters
    with open(ops.o.replace(".csv", "_clusters.csv"), "w", newline="") as f_clusters:
        clusters_fieldnames = [
            "event", "name", "cluster_num", "energy", "theta", "phi", "position", "num_hits"
        ]
        writer = csv.DictWriter(f_clusters, fieldnames=clusters_fieldnames)
        writer.writeheader()
        for cluster in all_clusters:
            writer.writerow(cluster)

    #Writing hits
    with open(ops.o.replace(".csv", "_hits.csv"), "w", newline="") as f_hits:
        hits_fieldnames = [
            "event", "cluster_num", "hit_num", "x", "y", "z", "energy", "time", "cell_id", "system", "layer"
        ]
        writer = csv.DictWriter(f_hits, fieldnames=hits_fieldnames)
        writer.writeheader()
        for hit in all_hits:
            writer.writerow(hit)

    #Writing unknown hits
    with open(ops.o.replace(".csv", "_unknown_hits.csv"), "w", newline="") as f_unknown:
        unknown_fieldnames = ["event", "hit_num", "energy", "cluster_num", "cell_id", "location", "layer", "position"]
        writer = csv.DictWriter(f_unknown, fieldnames=unknown_fieldnames)
        writer.writeheader()
        for unknown in unknown_hits:
            writer.writerow(unknown)

    #Writing PFOS 
    with open(ops.o.replace(".csv", "_pfos.csv"), "w", newline="") as f_pfos:
        pfos_fieldnames = [
            "event", "name", "pfo_num", "energy", "momentum", "pdg",
            "charge", "theta", "phi"
        ]
        writer = csv.DictWriter(f_pfos, fieldnames=pfos_fieldnames)
        writer.writeheader()
        for pfo in all_pfos:
            writer.writerow(pfo)
    print(f"Wrote {len(all_tracks)} tracks and {len(all_clusters)} clusters and {len(all_pfos)} pfos to CSV.")
#Write a success method 

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

if __name__ == "__main__":
    main()
        
            
