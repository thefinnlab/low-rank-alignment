import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import mne
    import h5py
    import torch
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from nilearn.plotting import plot_markers
    from mne_bids import BIDSPath

    from himalaya.backend import set_backend, get_backend
    from himalaya.ridge import RidgeCV
    from himalaya.scoring import correlation_score

    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return BIDSPath, get_backend, h5py, mne, np, pd, set_backend, torch


@app.cell
def _(set_backend, torch):
    if torch.cuda.is_available():
        set_backend("torch_cuda")
        print("Using cuda!")
    return


@app.cell
def _():
    import urllib.request
    import os

    bids_root = ""  # if using a local dataset, set this variable accordingly

    embedding_path = f"{bids_root}stimuli/gpt2-xl/features.hdf5"

    if not len(bids_root):
        url = f"https://s3.amazonaws.com/openneuro.org/ds005574/{embedding_path}"
        local_path = "features.hdf5"
    
        if not os.path.exists(local_path):
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, local_path)
            print("Download complete.")
        else:
            print("File already exists, skipping download.")
    
        embedding_path = local_path

    print(f"Using embedding file path: {embedding_path}")
    return bids_root, embedding_path, os, urllib


@app.cell
def _(embedding_path, h5py):
    modelname, layer = 'gpt2-xl', 24
    with h5py.File(embedding_path, "r") as f:
        contextual_embeddings = f[f"layer-{layer}"][...]
    print(f"LLM embedding matrix has shape: {contextual_embeddings.shape}")
    return (contextual_embeddings,)


@app.cell
def _(bids_root, os, pd, urllib):
    # Download the transcript, if required
    def _():
        transcript_path = f"{bids_root}stimuli/gpt2-xl/transcript.tsv"
        if not len(bids_root):
            url = f"https://s3.amazonaws.com/openneuro.org/ds005574/{transcript_path}"
            local_path = "transcript.tsv"
        
            if not os.path.exists(local_path):
                print(f"Downloading from {url}...")
                urllib.request.urlretrieve(url, local_path)
                print("Download complete.")
            else:
                print("File already exists, skipping download.")
        
            transcript_path = local_path
            return transcript_path

    transcript_path = _()

    # Load transcript
    df_contextual = pd.read_csv(transcript_path, sep="\t", index_col=0)
    if "rank" in df_contextual.columns:
        model_acc = (df_contextual["rank"] == 0).mean()
        print(f"Model accuracy: {model_acc*100:.3f}%")
    df_contextual.head()
    return (df_contextual,)


@app.cell
def _(contextual_embeddings, df_contextual, np):
    aligned_embeddings = []
    for _, group in df_contextual.groupby("word_idx"): # group by word index
        indices = group.index.to_numpy()
        average_emb = contextual_embeddings[indices].mean(0) # average features
        aligned_embeddings.append(average_emb)
    aligned_embeddings = np.stack(aligned_embeddings)
    print(f"LLM embeddings matrix has shape: {aligned_embeddings.shape}")
    return (aligned_embeddings,)


@app.cell
def _(df_contextual):
    df_word = df_contextual.groupby("word_idx").agg(dict(word="first", start="first", end="last"))
    df_word.head()
    return (df_word,)


@app.cell
def _(BIDSPath, bids_root, mne, os, urllib):
    def _():
        file_path = BIDSPath(root=f"{bids_root}derivatives/ecogprep",
                            subject="03", task="podcast", datatype="ieeg", description="highgamma",
                            suffix="ieeg", extension=".fif")
        print(f"File path within the dataset: {file_path}")
    
        # You only need to run this if using Colab (i.e. if you did not set bids_root to a local directory)
        if not len(bids_root):
            url = f"https://s3.amazonaws.com/openneuro.org/ds005574/{file_path}"
            if not os.path.exists(file_path.basename):
                print(f"Downloading from {url}...")
                urllib.request.urlretrieve(url, file_path.basename)
                print("Download complete.")
            else:
                print("File already exists, skipping download.")
            file_path = file_path.basename
        return file_path

    file_path = _()

    raw = mne.io.read_raw_fif(file_path, verbose=False)
    picks = mne.pick_channels_regexp(raw.ch_names, "LG[AB]*")
    raw = raw.pick(picks)
    raw
    return (raw,)


@app.cell
def _(df_word, np, raw):
    events = np.zeros((len(df_word), 3), dtype=int)
    events[:, 0] = (df_word.start * raw.info['sfreq']).astype(int)
    events.shape
    return (events,)


@app.cell
def _(events, mne, raw):
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-2.0,
        tmax=2.0,
        baseline=None,
        proj=False,
        event_id=None,
        preload=True,
        event_repeated="merge",
    )
    print(f"Epochs object has a shape of: {epochs._data.shape}")

    epochs = epochs.resample(sfreq=32, npad='auto', method='fft', window='hamming')
    print(f"Epochs object has a shape of: {epochs._data.shape}")
    return (epochs,)


@app.cell
def _():
    return


@app.cell
def _(epochs):
    epochs_data = epochs.get_data(copy=True)
    epochs_data = epochs_data.reshape(len(epochs), -1)
    print(f"ECoG data matrix shape: {epochs_data.shape}")
    return (epochs_data,)


@app.cell
def _(aligned_embeddings, df_word, epochs):
    selected_df = df_word.iloc[epochs.selection]
    averaged_embeddings = aligned_embeddings[epochs.selection]
    print(averaged_embeddings.shape)
    return (averaged_embeddings,)


@app.cell
def _(averaged_embeddings, epochs_data, get_backend, np):
    X = averaged_embeddings
    Y = epochs_data

    if "torch" in get_backend().__name__:
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)

    X.shape, Y.shape
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
