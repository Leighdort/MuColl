#!/usr/bin/env python3
"""
split_root_uproot.py
--------------------
Split a ROOT file into sub-files of N events each using uproot + awkward.

Requirements:
    pip install uproot awkward numpy

Usage:
    python split_root_uproot.py input.root --tree Events --chunk 10 --outdir ./split_files
"""

#This is an attempt to split the sim file
import argparse
import os
import uproot
import awkward as ak


def split_root_file(input_path, tree_name, chunk_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with uproot.open(input_path) as f:
        tree = f[tree_name]
        total_events = tree.num_entries
        print(f"Total events in '{tree_name}': {total_events}")

        n_files = (total_events + chunk_size - 1) // chunk_size
        print(f"Splitting into {n_files} files of up to {chunk_size} events each...")

        base_name = os.path.splitext(os.path.basename(input_path))[0]

        for i, batch in enumerate(tree.iterate(step_size=chunk_size, library="ak")):
            out_path = os.path.join(out_dir, f"{base_name}_part{i:05d}.root")
            with uproot.recreate(out_path) as out_file:
                out_file[tree_name] = batch
            n_events = len(batch[ak.fields(batch)[0]])
            print(f"  Written: {out_path}  ({n_events} events)")

    print(f"\nDone. {n_files} files written to '{out_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a ROOT file into chunks of N events.")
    parser.add_argument("input",          help="Path to input ROOT file")
    parser.add_argument("--tree",  "-t",  default="Events", help="TTree name (default: Events)")
    parser.add_argument("--chunk", "-n",  type=int, default=10, help="Events per output file (default: 10)")
    parser.add_argument("--outdir","-o",  default="./split_files", help="Output directory (default: ./split_files)")
    args = parser.parse_args()

    split_root_file(args.input, args.tree, args.chunk, args.outdir)
