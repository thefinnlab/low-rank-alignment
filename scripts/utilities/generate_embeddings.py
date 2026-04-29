"""
generate_embeddings.py — Produce per-token contextual embeddings for the podcast
transcript using a HuggingFace model, and save them in the same on-disk format
consumed by the ``transformer`` embedding getter.

Output (under ``{data_root}/stimuli/{output_name}/``):
  features.hdf5    — one dataset per retained layer, ``layer-{L}`` → (n_tokens, hidden_dim)
  transcript.tsv   — per-token rows: word_idx, word, start, end, hftoken, token_id
  metadata.json    — generation parameters (for reproducibility)

Generic causal generation processes ``context_window`` tokens and retains the
hidden states for the last ``stride`` positions (or all positions, on the first
window). For bidirectional models (RoBERTa), use ``attention_mode='bidirectional_strict'``
with ``stride=1`` so each target sees only preceding tokens. Podcast-compatible
causal LM generation uses ``attention_mode='podcast_per_token'`` and builds one
left-padded context row per target token, matching ``data/code/llm.py``.
"""

import argparse
import json
import os
import re

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from src.core.config import GenerateEmbeddingsConfig
from src.core.registry import get_dataset_adapter, resolve_dataset_context
from src.utils.config_utils import load_config, parse_override_args


_CAUSAL_MODEL_TYPES = {
    "gpt2",
    "gptj",
    "gpt_neox",
    "llama",
    "mistral",
    "falcon",
    "mpt",
    "bloom",
    "opt",
}
_BIDIRECTIONAL_MODEL_TYPES = {"bert", "roberta", "deberta", "deberta-v2", "electra"}


def _auto_output_name(hf_model_id: str) -> str:
    return hf_model_id.split("/")[-1]


def _tokenizer_name_or_path(cfg: GenerateEmbeddingsConfig) -> str:
    return cfg.tokenizer_name_or_path or cfg.hf_model_id


def _is_causal(model_type: str) -> bool:
    if model_type in _CAUSAL_MODEL_TYPES:
        return True
    if model_type in _BIDIRECTIONAL_MODEL_TYPES:
        return False
    raise ValueError(
        f"Unrecognized model_type={model_type!r}; set attention_mode explicitly in the config."
    )


def _resolve_defaults(cfg: GenerateEmbeddingsConfig, model_type: str):
    mode = cfg.attention_mode
    if mode == "auto":
        mode = "causal_native" if _is_causal(model_type) else "bidirectional_strict"

    if mode == "bidirectional_strict":
        stride = 1 if cfg.stride is None else cfg.stride
        assert stride == 1, (
            f"bidirectional_strict requires stride=1 (got {stride}); larger strides "
            "leak future tokens into target hidden states."
        )
    elif mode == "causal_native":
        stride = cfg.stride if cfg.stride is not None else cfg.context_window // 2
    elif mode == "podcast_per_token":
        assert cfg.context_window >= 1, "podcast_per_token requires context_window >= 1"
        stride = None
    elif mode == "bidirectional_masked":
        stride = cfg.stride if cfg.stride is not None else cfg.context_window // 2
    else:
        raise ValueError(f"Unknown attention_mode: {mode}")

    return mode, stride


def _resolve_dtype(dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def _resolve_np_dtype(dtype: str):
    return {"float32": np.float32, "float16": np.float16, "bfloat16": np.float32}[dtype]


def _load_model_for_generation(cfg: GenerateEmbeddingsConfig, model_cls):
    """Load a pretrained or randomly initialized model for embedding generation."""
    model_kwargs = dict(
        torch_dtype=_resolve_dtype(cfg.dtype),
        cache_dir=cfg.hf_cache_dir,
        local_files_only=cfg.local_files_only,
        token=cfg.hf_token,
    )
    if not cfg.random_init_model:
        model = model_cls.from_pretrained(cfg.hf_model_id, **model_kwargs)
    else:
        if cfg.random_init_seed is not None:
            torch.manual_seed(cfg.random_init_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.random_init_seed)
        config = AutoConfig.from_pretrained(
            cfg.hf_model_id,
            cache_dir=cfg.hf_cache_dir,
            local_files_only=cfg.local_files_only,
            token=cfg.hf_token,
        )
        model = model_cls.from_config(config, torch_dtype=_resolve_dtype(cfg.dtype))
    return model.to(cfg.device).eval()


def _select_generation_model_class(attention_mode: str):
    return AutoModelForCausalLM if attention_mode == "podcast_per_token" else AutoModel


def _tokenize_transcript(words: list[str], tokenizer):
    """Tokenize the pre-split word list and return (input_ids, word_ids).

    Uses ``is_split_into_words=True`` so every tokenizer family produces a canonical
    subtoken→word alignment via ``encoding.word_ids()``.
    """
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=False,
        return_tensors=None,
    )
    input_ids = np.asarray(enc["input_ids"], dtype=np.int64)
    word_ids = enc.word_ids()
    assert len(input_ids) == len(word_ids), "input_ids and word_ids length mismatch"
    mapped_words = {w for w in word_ids if w is not None}
    missing = set(range(len(words))) - mapped_words
    assert not missing, f"{len(missing)} words failed to tokenize: {sorted(missing)[:5]}..."
    return input_ids, word_ids


def _tokenize_transcript_podcast(words: list[str], tokenizer):
    """Tokenize words with the legacy podcast convention.

    ``data/code/llm.py`` tokenized each transcript word independently as
    ``tokenizer.tokenize(" " + word)``. This differs from
    ``is_split_into_words=True`` for GPT-style BPE tokenizers, so the parity
    path keeps this behavior explicit.
    """
    input_ids: list[int] = []
    word_ids: list[int] = []
    hftokens: list[str] = []
    for w_idx, word in enumerate(words):
        tokens = tokenizer.tokenize(" " + str(word))
        if not tokens:
            raise AssertionError(f"1 word failed to tokenize: {word!r}")
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        input_ids.extend(int(tid) for tid in token_ids)
        word_ids.extend([w_idx] * len(tokens))
        hftokens.extend(tokens)

    return np.asarray(input_ids, dtype=np.int64), word_ids, hftokens


def _build_transcript_df(
    words_df: pd.DataFrame,
    word_ids: list,
    input_ids: np.ndarray,
    tokenizer,
    hftokens: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the per-token transcript.tsv schema used by the embedding loaders."""
    if hftokens is None:
        hftokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
    rows = []
    for tok_idx, w_idx in enumerate(word_ids):
        src = words_df.iloc[w_idx]
        rows.append(
            dict(
                word_idx=int(w_idx),
                word=src["word"],
                start=float(src["start"]),
                end=float(src["end"]),
                hftoken=hftokens[tok_idx],
                token_id=int(input_ids[tok_idx]),
            )
        )
    return pd.DataFrame(rows)


def _resolve_layers(cfg_layers, num_hidden_states: int) -> list[int]:
    if cfg_layers == "all" or cfg_layers is None:
        return list(range(num_hidden_states))
    return [int(L) for L in cfg_layers]


def _load_generation_words(cfg: GenerateEmbeddingsConfig, output_name: str):
    context = resolve_dataset_context(
        dataset=cfg.dataset,
        subject_ids=[],
        per_dataset_config=cfg.per_dataset_config,
        data_root=cfg.data_root,
    )
    adapter = get_dataset_adapter(context.dataset)
    if context.dataset == "podcast":
        print(f"Loading transcript: {cfg.transcript_path}")
        words_df = pd.read_csv(cfg.transcript_path)
    else:
        print(f"Loading {context.dataset} transcript: {context.stimulus_id}")
        words_df = adapter.load_word_table(cfg.data_root, context)

    if "word" not in words_df.columns:
        raise ValueError("Transcript table must contain a 'word' column.")
    out_dir = adapter.get_embedding_dir(cfg.data_root, context, output_name)
    return words_df, context, out_dir


def _causal_window_plan(N: int, stride: int, T: int) -> list[tuple[int, int, int, int]]:
    """Plan the sliding windows for causal inference.

    Returns a list of ``(window_start, window_end, keep_lo, keep_hi)`` tuples whose
    keep ranges partition ``[0, T)`` exactly. The first window spans ``[0, min(N, T))``
    and retains every position (filling the initial context); subsequent windows span
    ``[end - N, end)`` and retain only the fresh positions ``[previous_end, end)``.
    Using ``keep_lo = previous_end`` (rather than ``end - stride``) keeps coverage
    disjoint at the ``T`` boundary when ``stride`` does not divide ``T - N`` evenly.
    """
    assert stride > 0, f"stride must be positive, got {stride}"
    assert stride <= N, f"stride ({stride}) must be <= context_window ({N})"
    plan: list[tuple[int, int, int, int]] = []
    p = 0
    is_first = True
    while p < T:
        end = min(p + N, T) if is_first else min(p + stride, T)
        start = max(0, end - N)
        keep_lo = start if is_first else p
        keep_hi = end
        plan.append((start, end, keep_lo, keep_hi))
        if end == T:
            break
        p = end
        is_first = False
    return plan


def _run_causal_sliding(
    model, input_ids: np.ndarray, layers: list[int], cfg, stride: int, out: dict
):
    """Sliding-window forward passes for causal/decoder models.

    Fills ``out[L]`` (n_tokens, hidden_dim) for each layer L. Causal attention
    enforces left-only naturally, so any stride is safe.
    """
    device = cfg.device
    N = cfg.context_window
    T = input_ids.shape[0]
    ids_t = torch.from_numpy(input_ids)

    plan = _causal_window_plan(N, stride, T)
    with tqdm(total=T, desc="tokens (causal)") as pbar:
        for start, end, keep_lo, keep_hi in plan:
            window = ids_t[start:end].unsqueeze(0).to(device)
            with torch.inference_mode():
                hs = model(window, output_hidden_states=True).hidden_states  # tuple of (1, w, d)
            for L in layers:
                out[L][keep_lo:keep_hi] = hs[L][0, keep_lo - start : keep_hi - start].float().cpu().numpy()
            pbar.update(keep_hi - keep_lo)


def _build_bidir_batch(
    input_ids: "torch.Tensor", targets: list[int], N: int, pad_id: int
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Build left-padded ``(batch_ids, batch_mask)`` for bidirectional_strict inference.

    For each target ``t`` in ``targets``, the window contains ``input_ids[max(0, t+1-N) : t+1]``
    right-aligned into a length-``N`` row, with ``pad_id`` filling the left padding.
    The attention mask is ``1`` over real tokens and ``0`` over padding, so padding
    tokens cannot be attended to and targets near the start see only the real
    context preceding them.
    """
    B = len(targets)
    batch_ids = torch.full((B, N), pad_id, dtype=torch.long)
    batch_mask = torch.zeros((B, N), dtype=torch.long)
    for row, tgt in enumerate(targets):
        lo = max(0, tgt + 1 - N)
        seq = input_ids[lo : tgt + 1]
        batch_ids[row, N - seq.shape[0] :] = seq
        batch_mask[row, N - seq.shape[0] :] = 1
    return batch_ids, batch_mask


def _build_podcast_context_batch(
    input_ids: "torch.Tensor", targets: list[int], context_len: int, pad_id: int
) -> "torch.Tensor":
    """Build podcast-style left-padded causal LM rows.

    For target token ``i``, the real sequence is
    ``input_ids[max(0, i - context_len): i + 1]`` and the row width is
    ``context_len + 1``. The target is always at ``-1`` and the logit position
    that predicts it is therefore ``-2``.
    """
    B = len(targets)
    width = context_len + 1
    batch_ids = torch.full((B, width), pad_id, dtype=torch.long)
    for row, tgt in enumerate(targets):
        lo = max(0, tgt - context_len)
        seq = input_ids[lo : tgt + 1]
        batch_ids[row, width - seq.shape[0] :] = seq
    return batch_ids


def _performance_from_logits(logits: "torch.Tensor", target_ids: "torch.Tensor"):
    """Return GPT-style prediction metrics for the target token."""
    probs = torch.softmax(logits.float(), dim=-1)
    true_probs = probs.gather(1, target_ids[:, None]).squeeze(1)
    top_pred = torch.argmax(probs, dim=-1)
    ranks = (probs > true_probs[:, None]).sum(dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(torch.finfo(probs.dtype).tiny))).sum(dim=-1)
    return {
        "rank": ranks.cpu().numpy().astype(np.int64),
        "true_prob": true_probs.cpu().numpy().astype(np.float64),
        "top_pred": top_pred.cpu().numpy().astype(np.int64),
        "entropy": entropy.cpu().numpy().astype(np.float64),
    }


def _run_bidirectional_strict(
    model, input_ids: np.ndarray, layers: list[int], cfg, tokenizer, out: dict
):
    """Per-target forward passes for bidirectional models with strict left-only context.

    For each target token position i, feed the last min(i+1, N) tokens up to and
    including i, and retain the hidden state at the target position. Batched for
    GPU efficiency — left-padded windows with an attention mask so padding tokens
    cannot be attended to.
    """
    device = cfg.device
    N = cfg.context_window
    T = input_ids.shape[0]
    ids_t = torch.from_numpy(input_ids)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    B = cfg.batch_size
    for batch_start in tqdm(range(0, T, B), desc="targets (bidir-strict)"):
        targets = list(range(batch_start, min(batch_start + B, T)))
        batch_ids, batch_mask = _build_bidir_batch(ids_t, targets, N, pad_id)
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)

        with torch.inference_mode():
            hs = model(
                batch_ids, attention_mask=batch_mask, output_hidden_states=True
            ).hidden_states  # tuple of (B, N, d)

        for L in layers:
            layer_hs = hs[L][:, -1, :].float().cpu().numpy()  # (B, d) — hidden state at each target
            for row, tgt in enumerate(targets):
                out[L][tgt] = layer_hs[row]


def _run_podcast_per_token(
    model, input_ids: np.ndarray, layers: list[int], cfg, tokenizer, out: dict
) -> pd.DataFrame:
    """Podcast-compatible per-token causal LM inference.

    Hidden states are retained from the target position (``-1``); next-token
    metrics are computed from logits at ``-2`` because those logits predict the
    target token at ``-1``.
    """
    device = cfg.device
    context_len = cfg.context_window
    T = input_ids.shape[0]
    ids_t = torch.from_numpy(input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    perf = {
        "rank": np.zeros(T, dtype=np.int64),
        "true_prob": np.zeros(T, dtype=np.float64),
        "top_pred": np.zeros(T, dtype=np.int64),
        "entropy": np.zeros(T, dtype=np.float64),
    }

    B = cfg.batch_size
    for batch_start in tqdm(range(0, T, B), desc="targets (podcast-causal)"):
        targets = list(range(batch_start, min(batch_start + B, T)))
        batch_ids = _build_podcast_context_batch(ids_t, targets, context_len, pad_id).to(device)
        target_ids = ids_t[targets].to(device)

        with torch.inference_mode():
            outputs = model(batch_ids, output_hidden_states=True)

        for L in layers:
            layer_hs = outputs.hidden_states[L][:, -1, :].float().cpu().numpy()
            for row, tgt in enumerate(targets):
                out[L][tgt] = layer_hs[row]

        batch_perf = _performance_from_logits(outputs.logits[:, -2, :], target_ids)
        for name, values in batch_perf.items():
            perf[name][batch_start : batch_start + len(targets)] = values

    return pd.DataFrame(perf)


def _write_features_hdf5(features_path: str, layers: list[int], out: dict):
    with h5py.File(features_path, "w") as f:
        for L in layers:
            f.create_dataset(f"layer-{L}", data=out[L], compression="gzip")


def generate(cfg: GenerateEmbeddingsConfig):
    output_name = cfg.output_name or _auto_output_name(cfg.hf_model_id)

    tokenizer_name_or_path = _tokenizer_name_or_path(cfg)
    print(f"Loading tokenizer: {tokenizer_name_or_path}")
    tok_kwargs = dict(
        cache_dir=cfg.hf_cache_dir,
        local_files_only=cfg.local_files_only,
        token=cfg.hf_token,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, add_prefix_space=cfg.add_prefix_space, **tok_kwargs
        )
    except TypeError:
        # Some tokenizers (e.g. SentencePiece/Llama) don't accept add_prefix_space.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tok_kwargs)

    model_cls = _select_generation_model_class(cfg.attention_mode)
    init_mode = "random init" if cfg.random_init_model else "pretrained"
    print(f"Loading model: {cfg.hf_model_id} ({init_mode}, dtype={cfg.dtype}, device={cfg.device})")
    model = _load_model_for_generation(cfg, model_cls)

    model_type = model.config.model_type
    attention_mode, stride = _resolve_defaults(cfg, model_type)
    print(f"model_type={model_type} attention_mode={attention_mode} stride={stride}")

    words_df, dataset_context, out_dir = _load_generation_words(cfg, output_name)
    words = words_df["word"].tolist()
    if attention_mode == "podcast_per_token":
        input_ids, word_ids, hftokens = _tokenize_transcript_podcast(words, tokenizer)
    else:
        input_ids, word_ids = _tokenize_transcript(words, tokenizer)
        hftokens = None
    T = input_ids.shape[0]
    hidden_dim = model.config.hidden_size
    num_hidden_states = model.config.num_hidden_layers + 1  # +1 for embedding layer
    layers = _resolve_layers(cfg.layers, num_hidden_states)
    print(f"n_tokens={T} hidden_dim={hidden_dim} layers_to_save={layers}")

    np_dtype = _resolve_np_dtype(cfg.output_dtype)
    out = {L: np.zeros((T, hidden_dim), dtype=np_dtype) for L in layers}

    performance_df = None
    if attention_mode == "causal_native":
        _run_causal_sliding(model, input_ids, layers, cfg, stride, out)
    elif attention_mode == "podcast_per_token":
        performance_df = _run_podcast_per_token(model, input_ids, layers, cfg, tokenizer, out)
    elif attention_mode == "bidirectional_strict":
        _run_bidirectional_strict(model, input_ids, layers, cfg, tokenizer, out)
    elif attention_mode == "bidirectional_masked":
        # Same sliding window as causal, but RoBERTa-style model with a causal mask.
        # Out-of-distribution for bidirectionally-trained encoders; documented as such.
        raise NotImplementedError(
            "bidirectional_masked is a documented escape hatch; implement when needed."
        )
    else:
        raise ValueError(f"Unknown attention_mode: {attention_mode}")

    transcript_df = _build_transcript_df(words_df, word_ids, input_ids, tokenizer, hftokens=hftokens)
    if performance_df is not None:
        transcript_df = pd.concat([transcript_df, performance_df], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    features_path = os.path.join(out_dir, "features.hdf5")
    print(f"Writing {features_path}")
    _write_features_hdf5(features_path, layers, out)

    transcript_path = os.path.join(out_dir, "transcript.tsv")
    print(f"Writing {transcript_path}")
    transcript_df.to_csv(transcript_path, sep="\t")

    if performance_df is not None:
        performance_path = os.path.join(out_dir, "performance.json")
        print(f"Writing {performance_path}")
        with open(performance_path, "w") as f:
            json.dump(performance_df.to_dict(orient="list"), f, indent=2)

    metadata = dict(
        hf_model_id=cfg.hf_model_id,
        output_name=output_name,
        model_type=model_type,
        hidden_dim=hidden_dim,
        num_hidden_states=num_hidden_states,
        layers_saved=layers,
        context_window=cfg.context_window,
        stride=stride,
        attention_mode=attention_mode,
        pooling=cfg.pooling,
        dtype=cfg.dtype,
        output_dtype=cfg.output_dtype,
        add_prefix_space=cfg.add_prefix_space,
        random_init_model=cfg.random_init_model,
        random_init_seed=cfg.random_init_seed,
        architecture_source=cfg.hf_model_id,
        n_tokens=T,
        n_words=len(words),
        transcript_path=cfg.transcript_path,
        dataset=dataset_context.dataset,
        data_root=cfg.data_root,
        stimulus_id=dataset_context.stimulus_id,
        per_dataset_config=cfg.per_dataset_config,
    )
    metadata_path = os.path.join(out_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {metadata_path}")


def main():
    # Use a local parser so we can accept --config multiple times (each runs
    # sequentially in the same job, reusing Python/CUDA setup). Overrides are
    # applied to every config in the run.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Path to YAML config file. Repeat to run multiple configs sequentially.",
    )
    args, unknown = parser.parse_known_args()
    overrides = parse_override_args(unknown)

    config_paths = args.config if args.config else [None]
    n = len(config_paths)
    for i, path in enumerate(config_paths, start=1):
        if n > 1:
            print(f"\n===== [{i}/{n}] Generating from {path or '<defaults>'} =====")
        cfg = load_config(path, overrides, GenerateEmbeddingsConfig)
        generate(cfg)


if __name__ == "__main__":
    main()
