#!/usr/bin/env python3

"""
Luminosity Stability Simulator
==============================

Run toy experiments to simulate the "luminosity stability" problem. Any
luminosity algorithm may in principle drift over time with respect to the "true"
luminosity, in addition to having its own run-to-run statistical fluctuations.
This script simulates these luminosity algorithms under a variety of scenarios
to study how best to extract a stability uncertainty.

Author: Joey Carter <joey.carter@cern.ch>
"""

from __future__ import absolute_import, division, print_function


import argparse
import os
import sys

import numpy as np
import pandas as pd

import ROOT as root
import atlasplots as aplt


def _docstring(docstring):
    """Return summary of docstring"""
    return " ".join(docstring.split("\n")[4:9]) if docstring else ""


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=_docstring(__doc__))
    parser.add_argument("--version", action="version", version="%(prog)s 0.1")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose messages; multiple -v result in more verbose messages",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=100,
        help="Number of runs to simulate (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--outdir", default="output", help="Path to output directory"
    )

    args = parser.parse_args()

    return args


def simulate(algos, args):
    """Run the simulation.

    Parameters
    ----------
    algos : dict
        Dictionary of lumi algorithms (defined in main).
    args : argparse Namespace
        Command line arguments from argparse.
    """
    # Index / run number: [1, 2, ..., 100]
    x = np.arange(1, args.n + 1)

    # Create dataframe to store true lumi and lumi algos per run
    df = pd.DataFrame(index=pd.Index(x, name="run"))

    # Normalized time
    df["time"] = x / args.n

    # True integrated luminosity per run
    # Make it increasing vs time and normalized such that integrated L over all runs = 1
    func = np.vectorize(lambda x: 1 - np.exp(-5 * x))
    df["truth"] = func(df["time"]) / np.sum(func(df["time"]))

    for algo, vals in algos.items():
        df[algo] = (
            df["truth"]
            * (1 + df["time"] * vals["drift"] / 100)
            * np.random.normal(1, vals["noise"] / 100, args.n)
        )

    if args.verbose >= 1:
        print("Data:")
        print(df)

    return df


def plot_raw(df, algos, args):
    """Plot the raw simulation data.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with per-run simulated data for each lumi algorithm.
    algos : dict
        Dictionary of lumi algorithms (defined in main).
    args : argparse Namespace
        Command line arguments from argparse.
    """
    fig, ax = aplt.subplots(name="raw", figsize=(800, 600))

    algo_names = sorted(list(set(df.columns).difference({"time"})))
    algo_names.insert(0, algo_names.pop(algo_names.index("truth")))

    ax.set_xlabel("Run number")
    ax.set_ylabel("Run-integrated luminosity [norm]")

    ax.text(0.2, 0.87, "#it{Lumi Simulation}")
    ax.text(0.2, 0.82, "Drift and noise w.r.t. #it{L}_{true}", size=20)

    for algo_name in algo_names:
        graph = root.TGraph(
            df.index.size,
            df.index.values.astype(np.float64),
            df[algo_name].values.astype(np.float64),
        )

        if algo_name == "truth":
            ax.plot(
                graph,
                "P",
                markercolor=root.kBlack,
                markerstyle=root.kFullCircle,
                label="Truth",
                labelfmt="P",
            )
        else:
            ax.plot(
                graph,
                "P",
                markercolor=algos[algo_name]["color"],
                markerstyle=algos[algo_name]["marker"],
                label=f"{algo_name}, drift: {algos[algo_name]['drift']:g}%, noise: {algos[algo_name]['noise']:g}%",
                labelfmt="P",
            )

    # Add margins around data
    ax.add_margins(top=0.15, left=0.05, right=0.05, bottom=0.05)

    # Add legend
    legend_height = 0.05 * len(algo_names)
    ax.legend(loc=(0.55, 0.50 - legend_height, 0.75, 0.50), textsize=20)

    # Save plot
    outfilename = (
        f"{args.outdir}/lumi_stability_simulator.n{df.index.size}.raw_values.pdf"
    )
    fig.savefig(outfilename)


def plot_stability(df, algos, args):
    """Plot the traditional stability plot for the simulation.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with per-run simulated data for each lumi algorithm.
    algos : dict
        Dictionary of lumi algorithms (defined in main).
    args : argparse Namespace
        Command line arguments from argparse.
    """
    # Plot vs run number and vs luminosity fraction
    indep_vars = [
        ("runnum", "Run number", df.index.values.astype(np.float64)),
        (
            "lumifrac",
            "Luminosity fraction (#it{L}_{true})",
            np.cumsum(df["truth"].values).astype(np.float64),
        ),
    ]

    for indep_var in indep_vars:
        # Create separate plot per denominator
        for denom in set(df.columns).difference({"time"}):
            # Select the set of algorithms for numerator
            algo_names = set(df.columns).difference({"time", "truth", denom})

            fig, ax = aplt.subplots(
                name=f"stability_by_{indep_var[0]}_wrt_{denom}", figsize=(800, 600)
            )
            ax.set_xlabel(indep_var[1])
            ax.set_ylabel(f"#it{{L}}_{{algo}}/#it{{L}}_{{{denom}}} #minus 1 [%]")

            ax.text(0.2, 0.87, "#it{Lumi Simulation}")
            ax.text(0.2, 0.82, "Drift and noise w.r.t. #it{L}_{true}", size=20)

            if denom != "truth":
                ax.text(
                    0.2,
                    0.77,
                    f"{denom}: drift: {algos[denom]['drift']:g}%, noise: {algos[denom]['noise']:g}%",
                    size=20,
                )

            # Draw line at y = 0
            line = root.TLine(ax.get_xlim()[0], 0, ax.get_xlim()[1], 0)
            ax.plot(line)

            for algo_name in sorted(list(algo_names)):
                graph = root.TGraph(
                    df.index.size,
                    indep_var[2],
                    ((df[algo_name] / df[denom] - 1) * 100).values.astype(np.float64),
                )
                ax.plot(
                    graph,
                    "P",
                    markercolor=algos[algo_name]["color"],
                    markerstyle=algos[algo_name]["marker"],
                    label=f"{algo_name}, drift: {algos[algo_name]['drift']:g}%, noise: {algos[algo_name]['noise']:g}%",
                    labelfmt="P",
                )

            ax.set_ylim(-2, 4)

            # Add margins around data
            ax.add_margins(left=0.05, right=0.05)

            # Update line at y = 0 after adding margins
            line.SetX1(ax.get_xlim()[0])
            line.SetX2(ax.get_xlim()[1])

            # Add legend
            legend_height = 0.05 * len(algo_names)
            ax.legend(loc=(0.55, 0.92 - legend_height, 0.75, 0.92), textsize=20)

            # Save plot
            outfilename = f"{args.outdir}/lumi_stability_simulator.n{df.index.size}.stability.by_{indep_var[0]}.wrt_{denom}.pdf"
            fig.savefig(outfilename)


def plot_lumi_weighted_hist(df, algos, args):
    """Plot the histograms of the luminosity-weighted lumi ratios to assess the
    mean and RMS for each algorithm w.r.t. each other algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame with per-run simulated data for each lumi algorithm.
    algos : dict
        Dictionary of lumi algorithms (defined in main).
    args : argparse Namespace
        Command line arguments from argparse.
    """
    # Create separate plot per denominator
    for denom in set(df.columns).difference({"time"}):
        # Select the set of algorithms for numerator
        algo_names = set(df.columns).difference({"time", "truth", denom})

        fig, ax = aplt.subplots(name=f"lumi_weighted_hist_{denom}", figsize=(800, 600))
        ax.set_xlabel(
            f"#it{{L}}-weighted #it{{L}}_{{algo}}/#it{{L}}_{{{denom}}} #minus 1 [%]"
        )
        ax.set_ylabel("Integrated luminosity [norm]")

        ax.text(0.2, 0.87, "#it{Lumi Simulation}")
        ax.text(0.2, 0.82, "Drift and noise w.r.t. #it{L}_{true}", size=20)

        if denom != "truth":
            ax.text(
                0.2,
                0.77,
                f"{denom}: drift: {algos[denom]['drift']:g}%, noise: {algos[denom]['noise']:g}%",
                size=20,
            )

        for i, algo_name in enumerate(sorted(list(algo_names))):
            # Fill histogram
            hist = root.TH1D(f"hist_{algo_name}_vs_{denom}", "", 40, -2, 2)
            for index, row in df.iterrows():
                hist.Fill((row[algo_name] / row[denom] - 1) * 100, row["truth"])

            ax.plot(
                hist,
                "",
                linecolor=algos[algo_name]["color"],
                linestyle=algos[algo_name]["line"],
                fillcolor=algos[algo_name]["color"],
                fillalpha=0.05,
                label=f"{algo_name}, drift: {algos[algo_name]['drift']:g}%, noise: {algos[algo_name]['noise']:g}%",
                labelfmt="L",
            )

            # Add labels with histogram mean +/- RMS
            label_yup = 0.65
            if i == 0:
                ax.text(0.2, label_yup + 0.05, "Mean #pm RMS", size=20)
            ax.text(
                0.2,
                label_yup - i * 0.05,
                f"{algo_name}: {hist.GetMean():.2g} #pm {hist.GetRMS():.2g} %",
                size=20,
            )

        # Add margins around data
        ax.add_margins(top=0.28)

        # Add legend
        legend_height = 0.05 * len(algo_names)
        ax.legend(loc=(0.55, 0.92 - legend_height, 0.75, 0.92), textsize=20)

        # Save plot
        outfilename = f"{args.outdir}/lumi_stability_simulator.n{df.index.size}.lumi_weighted_hist.wrt_{denom}.pdf"
        fig.savefig(outfilename)


def main():
    try:
        args = parse_args()

        # Define lumi algorithms
        # Format: {<algo name>: "drift": <drift wrt truth [%]>, "noise": <gaus noise [%]>}
        # Also include plot-formatting options (colour, line/marker style)
        # root.kBlack and root.kFullCircle are reserved for truth lumi
        algos = {
            "A": {
                "drift": 0.0,
                "noise": 0.1,
                "color": root.kAzure - 2,
                "marker": root.kOpenCircle,
                "line": 1,
            },
            "B": {
                "drift": 0.0,
                "noise": 0.3,
                "color": root.kOrange - 3,
                "marker": root.kOpenSquare,
                "line": 2,
            },
            "C": {
                "drift": 0.5,
                "noise": 0.1,
                "color": root.kGreen + 2,
                "marker": root.kOpenTriangleUp,
                "line": 3,
            },
            "D": {
                "drift": 0.5,
                "noise": 0.3,
                "color": root.kRed + 1,
                "marker": root.kOpenTriangleDown,
                "line": 4,
            },
            "E": {
                "drift": -1.0,
                "noise": 0.1,
                "color": root.kViolet + 1,
                "marker": root.kOpenDiamond,
                "line": 5,
            },
        }

        # Generate the simulated data and plot
        df = simulate(algos, args)

        # Set the ATLAS style
        aplt.set_atlas_style()

        # Create output directory
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)

        # Plot
        plot_raw(df, algos, args)
        plot_stability(df, algos, args)
        plot_lumi_weighted_hist(df, algos, args)

    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    root.gROOT.SetBatch()
    sys.exit(main())
