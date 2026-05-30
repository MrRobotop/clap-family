"""Regenerate the paper's dwell-vs-horizon and learned-gate figures (needs the [plots] extra)."""
from clap_family.experiments.reproduce import main

if __name__ == "__main__":
    main(make_plots=True)   # writes results.json + fig_*.png
