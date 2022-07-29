import numpy as np
import adios2 as ad2
import os
import argparse
import logging
import itertools

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="infile", nargs="+", type=str)
    parser.add_argument("--outfile", help="outfile", default="out.bp")
    args = parser.parse_args()

    fmt = "[%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)

    val_list = list()
    for fname in args.infile:
        logging.debug("Read: %s" % fname)
        with ad2.open(fname, "r") as f:
            val = f.read("X0")
            print(val.shape)
            val_list.append(val)

    out = np.array(val_list)
    out = np.moveaxis(out, 1, 2)
    print(out.shape)
    logging.debug("Write: %s" % args.outfile)

    with ad2.open(args.outfile, "w") as fw:
        shape = out.shape
        start = [
            0,
        ] * len(out.shape)
        count = out.shape
        fw.write("recon", out.copy(), shape, start, count)

    print("Done.")
