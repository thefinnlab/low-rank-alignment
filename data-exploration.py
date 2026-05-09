import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import h5py
    import os
    import matplotlib.pyplot as plt
    import json
    import pandas as pd
    from scipy import signal, stats
    import numpy as np
    import seaborn as sns
    import re

    return json, os, pd, re


@app.cell
def _():
    root_dir = "data/brain-treebank/"
    return (root_dir,)


@app.cell
def _(json, os, pd, re, root_dir):
    trial_metadata = []
    for trial_file in os.listdir(os.path.join(root_dir, f'subject_metadata')):
        match = re.search("sub_(\d+)_trial(\d+)_metadata.json", trial_file)
        if match:
            sub_id = int(match.group(1))
            trial_id = int(match.group(2))
            with open(os.path.join(root_dir, 'subject_metadata', trial_file), 'r') as _f:
                _metadata = json.load(_f)
                if "lanugage" in _metadata:
                    _metadata["language"] = _metadata.pop("lanugage")
                trial_metadata.append({"sub_id": sub_id, "trial_id": trial_id} | _metadata)
            
    trial_metadata_df = pd.DataFrame(trial_metadata)
    trial_metadata_df
    return (trial_metadata_df,)


@app.cell
def _(os, pd, root_dir, trial_metadata_df):
    # Path to transcript csv file
    # transcript_file_format = os.path.join(root_dir, f'transcripts/{movie_id}/features.csv')
    _movie_metadata = []
    for movie in os.listdir(os.path.join(root_dir, 'transcripts')):
        transcript_file = os.path.join(root_dir, f'transcripts/{movie}/features.csv')
        _df = pd.read_csv(transcript_file, index_col=0)
        _movie_metadata.append({"movie": movie, "length": _df["end"].max(), "num_words": _df.shape[0]})

    movie_metadata_df = pd.DataFrame(_movie_metadata)
    movie_metadata_df["word_rate"] = movie_metadata_df["num_words"] / movie_metadata_df["length"]
    _movie_viewings = trial_metadata_df.groupby("filename").size()
    movie_metadata_df["num_viewings"] = movie_metadata_df["movie"].map(_movie_viewings)
    movie_metadata_df
    return (movie_metadata_df,)


@app.cell
def _(movie_metadata_df):
    movie_metadata_df.sort_values("word_rate", ascending=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
