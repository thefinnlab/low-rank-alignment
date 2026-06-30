import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import hashlib
    import json
    from pathlib import Path

    import sys
    import os
    sys.path.insert(0, '/home/handrail6120/Documents/projects/alignment-attribution')

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.data_loaders import (
        epochs_from_cache,
        load_precomputed_epochs,
        read_electrode_file,
        read_subject_mapping,
    )

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        F,
        Path,
        SDPBackend,
        epochs_from_cache,
        hashlib,
        json,
        load_precomputed_epochs,
        mo,
        np,
        pd,
        plt,
        read_electrode_file,
        read_subject_mapping,
        sdpa_kernel,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Word-boundary Fisher features and ECoG correlations

    For a word ending at subtoken position $e$, this notebook computes the
    Fisher information of $p(x_{e+1}\mid x_{\leq e})$ with respect to the
    configured layer state $h_e$. Multi-token words therefore have the same
    feature definition and dimensionality as single-token words.

    Configure a Python-style layer range below. Stages can be run separately or
    in order with **Run all analyses**. Completed compatible results are skipped;
    check **Refresh** to recompute them. Fisher generation checkpoints to CSV.
    """)
    return


@app.cell
def _():
    # All paths are relative to the repository root unless made absolute.
    config = {
        "hf_model_id": "./gpt2-xl-local",
        "hf_cache_dir": None,
        "local_files_only": True,
        "data_root": "data",
        "transcript_path": "data/stimuli/podcast_transcript.csv",
        "output_root": "processed_data",
        "layer_min": 10,
        "layer_max": 48,
        "layer_step": 10,
        "method": "hutchinson",  # hutchinson | score_gradient
        "num_probes": 32,
        "num_samples": 256,
        "gradient_chunk_size": 8,
        "seed": 0,
        "context_window": 128,
        "stride": 64,
        "checkpoint_every": 25,
        "device": "cuda",
        "dtype": "float32",  # float32 | bfloat16 | float16
        "add_prefix_space": True,
        "subjects": list(range(1, 10)),
        "significant_channels_path": "processed_data/all_subject_sig.csv",
        "participant_map_path": "data/participants.tsv",
        "gpt2_transcript_path": "data/stimuli/gpt2-xl/transcript.tsv",
        "lag_tmin": -2.0,
        "lag_tmax": 2.0,
        "regression_type": "ridge",  # ols | ridge
        "num_folds": 5,
    }
    return (config,)


@app.cell(hide_code=True)
def _(np, torch):
    def resolve_dtype(dtype):
        try:
            return {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[dtype]
        except KeyError as error:
            raise ValueError(f"Unsupported dtype: {dtype}") from error

    def tokenize_transcript(words, tokenizer):
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors=None,
        )
        input_ids = np.asarray(encoding["input_ids"], dtype=np.int64)
        word_ids = encoding.word_ids()
        if len(input_ids) != len(word_ids):
            raise AssertionError("input_ids and word_ids have different lengths")
        mapped_words = {word_idx for word_idx in word_ids if word_idx is not None}
        missing_words = set(range(len(words))) - mapped_words
        if missing_words:
            raise ValueError(
                f"{len(missing_words)} transcript words failed to tokenize: "
                f"{sorted(missing_words)[:5]}"
            )
        return input_ids, word_ids

    def causal_window_plan(context_window, stride, n_tokens):
        if not 0 < stride <= context_window:
            raise ValueError("stride must be positive and no larger than context_window")
        plan = []
        previous_end = 0
        first_window = True
        while previous_end < n_tokens:
            if first_window:
                end = min(context_window, n_tokens)
            else:
                end = min(previous_end + stride, n_tokens)
            start = max(0, end - context_window)
            keep_lo = start if first_window else previous_end
            plan.append((start, end, keep_lo, end))
            if end == n_tokens:
                break
            previous_end = end
            first_window = False
        return plan

    return causal_window_plan, resolve_dtype, tokenize_transcript


@app.cell(hide_code=True)
def _(config, mo, pd):
    generation_button = mo.ui.run_button(label="Generate / resume Fisher features")
    correlation_button = mo.ui.run_button(label="Run ECoG correlations")
    model_fitting_button = mo.ui.run_button(label="Fit feature models")
    run_all_button = mo.ui.run_button(label="Run all analyses")
    refresh = mo.ui.checkbox(label="Refresh", value=False)
    mo.vstack(
        [
            mo.md("## Configuration"),
            pd.DataFrame(
                {"setting": list(config), "value": [repr(v) for v in config.values()]}
            ),
            mo.hstack(
                [
                    generation_button,
                    correlation_button,
                    model_fitting_button,
                    run_all_button,
                    refresh,
                ]
            ),
        ]
    )
    return (
        correlation_button,
        generation_button,
        model_fitting_button,
        refresh,
        run_all_button,
    )


@app.cell(hide_code=True)
def _(F, np, torch):
    def _validate_estimator_inputs(log_probs, hidden, target_pos):
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("Expected hidden state shape (1, sequence, hidden_size)")
        if not 0 <= target_pos < hidden.shape[1]:
            raise IndexError(f"target_pos={target_pos} is outside the hidden sequence")
        probabilities = log_probs.detach().exp()
        if not torch.isfinite(log_probs).all() or not torch.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite next-token probabilities")
        return probabilities

    def hutchinson_fisher(
        log_probs, hidden, target_pos, num_probes=32, seed=0
    ):
        """Estimate trace(F), trace(F^2), and PR with Rademacher probes."""
        if num_probes < 1:
            raise ValueError("num_probes must be positive")
        probabilities = _validate_estimator_inputs(log_probs, hidden, target_pos)

        auxiliary = torch.zeros_like(log_probs, requires_grad=True)
        jacobian_t_auxiliary = torch.autograd.grad(
            log_probs,
            hidden,
            grad_outputs=auxiliary,
            create_graph=True,
            retain_graph=True,
        )[0]

        generator = torch.Generator(device=hidden.device).manual_seed(seed)
        trace_samples = []
        trace_square_samples = []
        for _probe in range(num_probes):
            direction_at_target = torch.randint(
                0,
                2,
                (hidden.shape[-1],),
                generator=generator,
                device=hidden.device,
                dtype=torch.int8,
            ).to(hidden.dtype).mul_(2).sub_(1)
            direction = torch.zeros_like(hidden)
            direction[0, target_pos] = direction_at_target

            jacobian_direction = torch.autograd.grad(
                jacobian_t_auxiliary,
                auxiliary,
                grad_outputs=direction,
                retain_graph=True,
            )[0]
            fisher_direction = torch.autograd.grad(
                log_probs,
                hidden,
                grad_outputs=probabilities * jacobian_direction,
                retain_graph=True,
            )[0][0, target_pos].float()

            if not torch.isfinite(fisher_direction).all():
                raise FloatingPointError(
                    "Non-finite Fisher-vector product; use float32 or bfloat16"
                )
            trace_samples.append(direction_at_target.float() @ fisher_direction)
            trace_square_samples.append(fisher_direction.square().sum())

        trace = torch.stack(trace_samples).mean()
        trace_square = torch.stack(trace_square_samples).mean()
        if not torch.isfinite(trace) or trace <= 0:
            raise FloatingPointError(
                "Hutchinson produced a nonpositive trace; increase num_probes"
            )
        if not torch.isfinite(trace_square) or trace_square <= 0:
            raise FloatingPointError("Invalid Hutchinson estimate of trace(F^2)")
        pr = trace.square() / trace_square
        return {
            "pr": float(pr.detach().cpu()),
            "trace": float(trace.detach().cpu()),
            "trace_square": float(trace_square.detach().cpu()),
        }

    def score_gradient_fisher(
        log_probs,
        hidden,
        target_pos,
        num_samples=32,
        gradient_chunk_size=8,
        seed=0,
    ):
        """Estimate Fisher moments from score gradients and U-statistics."""
        if num_samples < 2:
            raise ValueError("num_samples must be at least two")
        if gradient_chunk_size < 1:
            raise ValueError("gradient_chunk_size must be positive")
        probabilities = _validate_estimator_inputs(log_probs, hidden, target_pos)
        generator = torch.Generator(device=hidden.device).manual_seed(seed)
        classes = torch.multinomial(
            probabilities,
            num_samples,
            replacement=True,
            generator=generator,
        )

        gradient_chunks = []
        for start in range(0, num_samples, gradient_chunk_size):
            selected = classes[start : start + gradient_chunk_size]
            grad_outputs = F.one_hot(
                selected, num_classes=log_probs.numel()
            ).to(log_probs.dtype)
            gradients = torch.autograd.grad(
                log_probs,
                hidden,
                grad_outputs=grad_outputs,
                is_grads_batched=True,
                retain_graph=True,
            )[0][:, 0, target_pos].float()
            gradient_chunks.append(gradients.detach())

        gradients = torch.cat(gradient_chunks)
        if not torch.isfinite(gradients).all():
            raise FloatingPointError("Non-finite sampled score gradients")
        gram = gradients @ gradients.T
        squared_norms = gram.diagonal()
        denominator = num_samples * (num_samples - 1)
        trace = squared_norms.mean()
        trace_squared = (
            squared_norms.sum().square() - squared_norms.square().sum()
        ) / denominator
        trace_square = (
            gram.square().sum() - gram.diagonal().square().sum()
        ) / denominator
        if trace <= 0 or trace_squared <= 0 or trace_square <= 0:
            raise FloatingPointError(
                "Degenerate score-gradient estimate; increase num_samples"
            )
        pr = trace_squared / trace_square
        return {
            "pr": float(pr.detach().cpu()),
            "trace": float(trace.detach().cpu()),
            "trace_square": float(trace_square.detach().cpu()),
        }

    def validate_fisher_estimators(hutchinson_fisher, score_gradient_fisher):
        """Cheap explicit-Fisher smoke test used before transcript generation."""
        hidden = torch.tensor([[[0.3, -0.2, 0.1]]], requires_grad=True)
        weights = torch.tensor(
            [[2.0, -0.5, 0.1], [-0.3, 1.2, 0.7], [0.4, -0.8, 1.5]]
        )
        log_probs = torch.log_softmax(weights @ hidden[0, 0], dim=0)
        probabilities = log_probs.detach().exp()
        scores = weights - probabilities @ weights
        fisher = scores.T @ (probabilities[:, None] * scores)
        exact_trace = float(torch.trace(fisher))
        exact_pr = float(torch.trace(fisher).square() / fisher.square().sum())

        hutchinson = hutchinson_fisher(
            log_probs, hidden, 0, num_probes=2048, seed=7
        )
        score = score_gradient_fisher(
            log_probs,
            hidden,
            0,
            num_samples=2048,
            gradient_chunk_size=128,
            seed=7,
        )
        for name, estimate in (("hutchinson", hutchinson), ("score", score)):
            if not np.isclose(estimate["trace"], exact_trace, rtol=0.15):
                raise AssertionError(f"{name} trace failed synthetic validation")
            if not np.isclose(estimate["pr"], exact_pr, rtol=0.20):
                raise AssertionError(f"{name} PR failed synthetic validation")

    return hutchinson_fisher, score_gradient_fisher, validate_fisher_estimators


@app.cell(hide_code=True)
def _(Path, config, hashlib, json):
    def configured_layers():
        step = config["layer_step"]
        if step <= 0:
            raise ValueError("layer_step must be positive")
        layers = list(range(config["layer_min"], config["layer_max"], step))
        layers += [config["layer_max"]]
        if not layers:
            raise ValueError("The configured layer range is empty")
        return layers

    def model_slug():
        model_id = str(config["hf_model_id"]).rstrip("/")
        if model_id.startswith(("./", "../", "/")):
            return Path(model_id).name
        return model_id.replace("/", "--")

    def layer_paths(layer):
        root = Path(config["output_root"]) / model_slug() / f"layer-{layer}"
        return {
            "fisher_dir": root / "fisher_features",
            "fisher_csv": root / "fisher_features" / "features.csv",
            "fisher_metadata": root / "fisher_features" / "metadata.json",
            "ecog_dir": root / "ecog_correlations",
            "ecog_npz": root / "ecog_correlations" / "correlations.npz",
            "ecog_metadata": root / "ecog_correlations" / "metadata.json",
            "models_dir": root / "feature_models",
            "models_npz": root / "feature_models" / "correlations.npz",
            "models_metadata": root / "feature_models" / "metadata.json",
        }

    def make_signature(stage, payload):
        body = {"stage": stage, **payload}
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest, body

    def file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    return configured_layers, file_sha256, layer_paths, make_signature


@app.cell(hide_code=True)
def _(np):
    def columnwise_pearson(feature, neural):
        feature = np.asarray(feature, dtype=np.float64)
        neural = np.asarray(neural, dtype=np.float64)
        centered_feature = feature - feature.mean()
        centered_neural = neural - neural.mean(axis=0, keepdims=True)
        numerator = np.einsum("n,nel->el", centered_feature, centered_neural)
        denominator = np.sqrt(
            np.square(centered_feature).sum()
            * np.square(centered_neural).sum(axis=0)
        )
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )

    return (columnwise_pearson,)


@app.cell(hide_code=True)
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    SDPBackend,
    causal_window_plan,
    config,
    configured_layers,
    file_sha256,
    hutchinson_fisher,
    json,
    layer_paths,
    make_signature,
    np,
    pd,
    resolve_dtype,
    score_gradient_fisher,
    sdpa_kernel,
    tokenize_transcript,
    torch,
    tqdm,
    validate_fisher_estimators,
):
    def run_fisher_batch(refresh=False):
        if config["method"] not in {"hutchinson", "score_gradient"}:
            raise ValueError("Unsupported Fisher method")
        if not 0 < config["stride"] <= config["context_window"]:
            raise ValueError("stride must be in [1, context_window]")
        if config["checkpoint_every"] < 1:
            raise ValueError("checkpoint_every must be positive")
        if config["device"].startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        layers = configured_layers()
        words_df = pd.read_csv(config["transcript_path"])
        required = {"word", "start", "end"}
        if not required.issubset(words_df):
            raise ValueError(f"Transcript must contain {required}")

        signature_keys = (
            "hf_model_id", "transcript_path", "method", "num_probes",
            "num_samples", "gradient_chunk_size", "seed", "context_window",
            "stride", "dtype", "add_prefix_space",
        )
        signatures = {}
        statuses = {}
        active_layers = []
        rows_by_layer = {}
        completed_by_layer = {}
        for layer in layers:
            payload = {key: config[key] for key in signature_keys}
            payload["layer"] = layer
            signature, signature_payload = make_signature("fisher", payload)
            signatures[layer] = (signature, signature_payload)
            paths = layer_paths(layer)
            complete = False
            if not refresh and paths["fisher_csv"].exists() and paths["fisher_metadata"].exists():
                try:
                    metadata = json.loads(paths["fisher_metadata"].read_text())
                    existing = pd.read_csv(paths["fisher_csv"])
                    complete = (
                        metadata.get("complete") is True
                        and metadata.get("signature") == signature
                        and metadata.get("feature_sha256")
                        == file_sha256(paths["fisher_csv"])
                        and len(existing) == len(words_df)
                        and existing["word_idx"].is_unique
                        and set(("pr", "trace", "log_trace", "layer")).issubset(existing)
                        and existing["layer"].eq(layer).all()
                        and np.isfinite(existing[["pr", "trace", "log_trace"]]).all().all()
                    )
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    complete = False
            if complete:
                statuses[layer] = "skipped (complete)"
                continue

            paths["fisher_dir"].mkdir(parents=True, exist_ok=True)
            rows = []
            if refresh:
                paths["fisher_csv"].unlink(missing_ok=True)
            if not refresh and paths["fisher_csv"].exists() and paths["fisher_metadata"].exists():
                try:
                    metadata = json.loads(paths["fisher_metadata"].read_text())
                    if metadata.get("signature") == signature:
                        checkpoint = pd.read_csv(paths["fisher_csv"])
                        if checkpoint["word_idx"].is_unique:
                            rows = checkpoint.to_dict("records")
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    rows = []
            rows_by_layer[layer] = rows
            completed_by_layer[layer] = {int(row["word_idx"]) for row in rows}
            active_layers.append(layer)
            statuses[layer] = "resumed" if rows else "generated"

        if not active_layers:
            return statuses

        validate_fisher_estimators(hutchinson_fisher, score_gradient_fisher)
        tokenizer_kwargs = {
            "cache_dir": config["hf_cache_dir"],
            "local_files_only": config["local_files_only"],
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config["hf_model_id"],
                add_prefix_space=config["add_prefix_space"],
                **tokenizer_kwargs,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                config["hf_model_id"], **tokenizer_kwargs
            )
        input_ids, word_ids = tokenize_transcript(words_df["word"].tolist(), tokenizer)
        word_positions = {}
        for token_position, word_idx in enumerate(word_ids):
            word_positions.setdefault(int(word_idx), []).append(token_position)
        target_by_position = {
            positions[-1]: word_idx for word_idx, positions in word_positions.items()
        }

        model = AutoModelForCausalLM.from_pretrained(
            config["hf_model_id"],
            cache_dir=config["hf_cache_dir"],
            local_files_only=config["local_files_only"],
            torch_dtype=resolve_dtype(config["dtype"]),
        ).to(config["device"]).eval()
        num_hidden_states = model.config.num_hidden_layers + 1
        invalid = [layer for layer in layers if not 0 <= layer < num_hidden_states]
        if invalid:
            raise ValueError(
                f"Layers {invalid} are outside [0, {num_hidden_states - 1}]"
            )

        plan = causal_window_plan(
            config["context_window"], config["stride"], len(input_ids)
        )
        ids_tensor = torch.from_numpy(input_ids)
        total_remaining = sum(
            len(words_df) - len(completed_by_layer[layer]) for layer in active_layers
        )
        for layer in active_layers:
            signature, signature_payload = signatures[layer]
            metadata = {
                "complete": False,
                "signature": signature,
                "generation_config": signature_payload,
                "model_type": model.config.model_type,
                "num_hidden_states": num_hidden_states,
                "n_words": len(words_df),
                "n_tokens": len(input_ids),
            }
            layer_paths(layer)["fisher_metadata"].write_text(json.dumps(metadata, indent=2))

        with tqdm(total=total_remaining, desc="layer-word Fishers") as progress:
            for start, end, keep_lo, keep_hi in plan:
                targets = [
                    position for position in target_by_position
                    if keep_lo <= position < keep_hi
                    and any(
                        target_by_position[position] not in completed_by_layer[layer]
                        for layer in active_layers
                    )
                ]
                if not targets:
                    continue
                window = ids_tensor[start:end].unsqueeze(0).to(config["device"])
                with sdpa_kernel(SDPBackend.MATH):
                    outputs = model(
                        window, output_hidden_states=True, use_cache=False,
                        return_dict=True,
                    )
                for global_position in sorted(targets):
                    local_position = global_position - start
                    word_idx = target_by_position[global_position]
                    log_probs = torch.log_softmax(
                        outputs.logits[0, local_position].float(), dim=-1
                    )
                    word_seed = config["seed"] + word_idx
                    for layer in active_layers:
                        if word_idx in completed_by_layer[layer]:
                            continue
                        hidden = outputs.hidden_states[layer]
                        if config["method"] == "hutchinson":
                            estimate = hutchinson_fisher(
                                log_probs, hidden, local_position,
                                num_probes=config["num_probes"], seed=word_seed,
                            )
                            estimator_count = config["num_probes"]
                        else:
                            estimate = score_gradient_fisher(
                                log_probs, hidden, local_position,
                                num_samples=config["num_samples"],
                                gradient_chunk_size=config["gradient_chunk_size"],
                                seed=word_seed,
                            )
                            estimator_count = config["num_samples"]
                        source = words_df.iloc[word_idx]
                        rows_by_layer[layer].append({
                            "word_idx": word_idx,
                            "word": source["word"],
                            "start": float(source["start"]),
                            "end": float(source["end"]),
                            "final_token_idx": global_position,
                            "final_hftoken": tokenizer.convert_ids_to_tokens(
                                int(input_ids[global_position])
                            ),
                            "token_count": len(word_positions[word_idx]),
                            "pr": estimate["pr"],
                            "trace": estimate["trace"],
                            "log_trace": float(np.log(estimate["trace"])),
                            "trace_square": estimate["trace_square"],
                            "method": config["method"],
                            "layer": layer,
                            "estimator_count": estimator_count,
                            "seed": word_seed,
                        })
                        completed_by_layer[layer].add(word_idx)
                        progress.update(1)
                        if len(rows_by_layer[layer]) % config["checkpoint_every"] == 0:
                            pd.DataFrame(rows_by_layer[layer]).sort_values("word_idx").to_csv(
                                layer_paths(layer)["fisher_csv"], index=False
                            )
                del outputs, window
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for layer in active_layers:
            paths = layer_paths(layer)
            features = pd.DataFrame(rows_by_layer[layer]).sort_values("word_idx")
            if len(features) != len(words_df) or not features["word_idx"].is_unique:
                raise AssertionError(f"Layer {layer} feature table is incomplete")
            if not np.isfinite(features[["pr", "trace", "log_trace"]]).all().all():
                raise FloatingPointError(f"Layer {layer} contains non-finite features")
            features.to_csv(paths["fisher_csv"], index=False)
            metadata = json.loads(paths["fisher_metadata"].read_text())
            metadata.update({
                "complete": True,
                "feature_sha256": file_sha256(paths["fisher_csv"]),
            })
            paths["fisher_metadata"].write_text(json.dumps(metadata, indent=2))
        return statuses

    return (run_fisher_batch,)


@app.cell(hide_code=True)
def _(
    columnwise_pearson,
    config,
    configured_layers,
    epochs_from_cache,
    file_sha256,
    json,
    layer_paths,
    load_precomputed_epochs,
    make_signature,
    np,
    pd,
    read_electrode_file,
    read_subject_mapping,
):
    def run_ecog_batch(refresh=False):
        from wordfreq import word_frequency, zipf_frequency

        stored_names = ("pr", "trace", "log_trace")
        feature_names = (
            *stored_names, "pr_x_log_trace", "surprisal", "entropy",
            "word_frequency", "word_zipf_frequency",
        )
        tokens = pd.read_csv(config["gpt2_transcript_path"], sep="\t", index_col=0)
        if not {"word_idx", "true_prob", "entropy"}.issubset(tokens):
            raise ValueError("GPT-2 transcript lacks word_idx, true_prob, or entropy")
        true_prob = pd.to_numeric(tokens["true_prob"], errors="coerce")
        token_entropy = pd.to_numeric(tokens["entropy"], errors="coerce")
        if not np.isfinite(true_prob).all() or (true_prob <= 0).any():
            raise ValueError("Invalid true-token probabilities")
        if not np.isfinite(token_entropy).all():
            raise ValueError("Invalid token entropies")
        word_surprisal = tokens.assign(value=-np.log(true_prob)).groupby("word_idx")["value"].sum()
        word_entropy = tokens.assign(value=token_entropy).groupby("word_idx")["value"].sum()
        subject_mapping = (
            read_subject_mapping(config["participant_map_path"])
            if config["participant_map_path"] else None
        )
        significant = read_electrode_file(
            config["significant_channels_path"], subject_mapping=subject_mapping
        )
        statuses = {}
        for layer in configured_layers():
            paths = layer_paths(layer)
            if not paths["fisher_csv"].exists() or not paths["fisher_metadata"].exists():
                raise FileNotFoundError(f"Layer {layer} has no complete Fisher results")
            fisher_metadata = json.loads(paths["fisher_metadata"].read_text())
            if not fisher_metadata.get("complete"):
                raise RuntimeError(f"Layer {layer} Fisher results are incomplete")
            fisher_hash = file_sha256(paths["fisher_csv"])
            payload = {
                "layer": layer,
                "fisher_sha256": fisher_hash,
                "subjects": config["subjects"],
                "data_root": config["data_root"],
                "significant_channels_path": config["significant_channels_path"],
                "participant_map_path": config["participant_map_path"],
                "gpt2_transcript_path": config["gpt2_transcript_path"],
                "lag_tmin": config["lag_tmin"],
                "lag_tmax": config["lag_tmax"],
                "feature_names": feature_names,
            }
            signature, signature_payload = make_signature("ecog_correlations", payload)
            complete = False
            if not refresh and paths["ecog_npz"].exists() and paths["ecog_metadata"].exists():
                try:
                    metadata = json.loads(paths["ecog_metadata"].read_text())
                    with np.load(paths["ecog_npz"], allow_pickle=False) as saved:
                        correlations = saved["correlations"]
                        complete = (
                            metadata.get("complete") is True
                            and metadata.get("signature") == signature
                            and correlations.ndim == 3
                            and correlations.shape[0] == len(feature_names)
                            and correlations.shape[1] == len(saved["electrode_names"])
                            and correlations.shape[1] == len(saved["electrode_subjects"])
                            and correlations.shape[2] == len(saved["lags"])
                            and np.array_equal(saved["feature_names"], feature_names)
                            and metadata.get("shape") == list(correlations.shape)
                        )
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    complete = False
            if complete:
                statuses[layer] = "skipped (complete)"
                continue

            features = pd.read_csv(paths["fisher_csv"]).sort_values("word_idx")
            if not features["word_idx"].is_unique:
                raise ValueError("Fisher features must have unique word_idx values")
            features = features.set_index("word_idx", drop=False)
            unique_words = features["word"].astype(str).unique()
            features["word_frequency"] = features["word"].astype(str).map({
                word: word_frequency(word, "en", minimum=1e-9) for word in unique_words
            })
            features["word_zipf_frequency"] = features["word"].astype(str).map({
                word: zipf_frequency(word, "en", minimum=1.0) for word in unique_words
            })
            subject_arrays = []
            electrode_subjects = []
            electrode_names = []
            subject_word_counts = {}
            skipped_channels = 0
            lags = None
            summary_rows = []
            for subject in config["subjects"]:
                if subject not in significant:
                    continue
                cache = load_precomputed_epochs(subject, config["data_root"])
                usable = [ch for ch in significant[subject] if ch in cache["ch_names"]]
                skipped_channels += len(significant[subject]) - len(usable)
                if not usable:
                    continue
                selection = cache["selection"].astype(int)
                missing = set(selection) - set(features.index)
                if missing:
                    raise ValueError(f"Layer {layer}, subject {subject}: missing Fisher words")
                neural = epochs_from_cache(cache, elec_names=usable)
                subject_lags = np.linspace(
                    config["lag_tmin"], config["lag_tmax"], cache["n_lags"],
                    endpoint=False,
                )
                if lags is None:
                    lags = subject_lags
                elif not np.allclose(lags, subject_lags):
                    raise ValueError("Subjects have inconsistent lag arrays")
                sf = features.loc[selection]
                pr = sf["pr"].to_numpy()
                log_trace = sf["log_trace"].to_numpy()
                values = {name: sf[name].to_numpy() for name in stored_names}
                values.update({
                    "pr_x_log_trace": (pr - pr.mean()) * (log_trace - log_trace.mean()),
                    "surprisal": word_surprisal.loc[selection].to_numpy(),
                    "entropy": word_entropy.loc[selection].to_numpy(),
                    "word_frequency": sf["word_frequency"].to_numpy(),
                    "word_zipf_frequency": sf["word_zipf_frequency"].to_numpy(),
                })
                subject_corrs = np.stack([
                    columnwise_pearson(values[name], neural) for name in feature_names
                ])
                subject_arrays.append(subject_corrs)
                electrode_subjects.extend([subject] * len(usable))
                electrode_names.extend(map(str, usable))
                subject_word_counts[str(subject)] = len(selection)
                for feature_idx, feature_name in enumerate(feature_names):
                    for electrode_idx, electrode in enumerate(usable):
                        for lag_idx, lag in enumerate(lags):
                            summary_rows.append({
                                "subject": subject,
                                "electrode": electrode,
                                "feature": feature_name,
                                "lag": float(lag),
                                "correlation": float(subject_corrs[feature_idx, electrode_idx, lag_idx]),
                            })
            if not subject_arrays:
                raise ValueError("No subjects with usable significant electrodes")
            correlations = np.concatenate(subject_arrays, axis=1)
            if correlations.shape != (len(feature_names), len(electrode_names), len(lags)):
                raise AssertionError("Unexpected ECoG correlation shape")
            paths["ecog_dir"].mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                paths["ecog_npz"], correlations=correlations,
                feature_names=np.asarray(feature_names),
                electrode_subjects=np.asarray(electrode_subjects),
                electrode_names=np.asarray(electrode_names), lags=lags,
            )
            correlation_df = pd.DataFrame(summary_rows)
            def pooled(group):
                values = group["correlation"].dropna().clip(-0.999999, 0.999999)
                return pd.Series({
                    "n_electrodes": len(values),
                    "median_r": values.median(),
                    "fisher_z_mean_r": np.tanh(np.arctanh(values).mean()),
                    "q25_r": values.quantile(0.25), "q75_r": values.quantile(0.75),
                })
            joint = correlation_df.groupby(["feature", "lag"], sort=False).apply(
                pooled, include_groups=False
            ).reset_index()
            by_subject = correlation_df.groupby(
                ["subject", "feature", "lag"], sort=False
            ).apply(pooled, include_groups=False).reset_index()
            joint.to_csv(paths["ecog_dir"] / "joint_summary.csv", index=False)
            by_subject.to_csv(paths["ecog_dir"] / "subject_summary.csv", index=False)
            metadata = {
                "complete": True, "signature": signature,
                "analysis_config": signature_payload,
                "shape": list(correlations.shape),
                "axes": ["feature", "electrode", "lag"],
                "subject_word_counts": subject_word_counts,
                "skipped_channels": skipped_channels,
            }
            paths["ecog_metadata"].write_text(json.dumps(metadata, indent=2))
            statuses[layer] = "computed"
        return statuses

    return (run_ecog_batch,)


@app.cell(hide_code=True)
def _(
    config,
    configured_layers,
    epochs_from_cache,
    file_sha256,
    json,
    layer_paths,
    load_precomputed_epochs,
    make_signature,
    np,
    pd,
    read_electrode_file,
    read_subject_mapping,
):
    def run_feature_models_batch(refresh=False):
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from wordfreq import word_frequency, zipf_frequency
        from src.utils.encoding import get_ridge_model_constructor, train_encoding

        feature_sets = {
            "log word frequency + surprisal + entropy": ["word_zipf_frequency", "surprisal", "entropy"],
            "PR": ["pr"],
            "log trace": ["log_trace"],
            "PR x log trace": ["pr_x_log_trace"],
            "PR + log trace": ["pr", "log_trace"],
            "PR + log trace + interaction": ["pr", "log_trace", "pr_x_log_trace"],
            "PR + surprisal + log freq + entropy": ["pr", "surprisal", "word_zipf_frequency", "entropy"],
            "log trace + surprisal + log freq + entropy": ["log_trace", "surprisal", "word_zipf_frequency", "entropy"],
            "interaction + surprisal + log freq + entropy": ["pr_x_log_trace", "surprisal", "word_zipf_frequency", "entropy"],
            "PR + log trace + surprisal + log freq + entropy": ["pr", "log_trace", "surprisal", "word_zipf_frequency", "entropy"],
            "PR + log trace + interaction + surprisal": ["pr", "log_trace", "pr_x_log_trace", "surprisal"],
            "PR + log trace + interaction + surprisal + log freq": ["pr", "log_trace", "pr_x_log_trace", "surprisal", "word_zipf_frequency"],
            "PR + log trace + interaction + surprisal + log freq + entropy": ["pr", "log_trace", "pr_x_log_trace", "surprisal", "word_zipf_frequency", "entropy"],
        }
        regression_type = config["regression_type"]
        num_folds = config["num_folds"]
        if regression_type == "ridge":
            model_constructor = get_ridge_model_constructor()
        elif regression_type == "ols":
            def model_constructor(_fold):
                return make_pipeline(StandardScaler(), LinearRegression(fit_intercept=True))
        else:
            raise ValueError("regression_type must be 'ols' or 'ridge'")

        tokens = pd.read_csv(config["gpt2_transcript_path"], sep="\t", index_col=0)
        true_prob = pd.to_numeric(tokens["true_prob"], errors="coerce")
        token_entropy = pd.to_numeric(tokens["entropy"], errors="coerce")
        if not np.isfinite(true_prob).all() or (true_prob <= 0).any():
            raise ValueError("Invalid true-token probabilities")
        if not np.isfinite(token_entropy).all():
            raise ValueError("Invalid token entropies")
        word_surprisal = tokens.assign(value=-np.log(true_prob)).groupby("word_idx")["value"].sum()
        word_entropy = tokens.assign(value=token_entropy).groupby("word_idx")["value"].sum()
        subject_mapping = (
            read_subject_mapping(config["participant_map_path"])
            if config["participant_map_path"] else None
        )
        significant = read_electrode_file(
            config["significant_channels_path"], subject_mapping=subject_mapping
        )
        statuses = {}
        for layer in configured_layers():
            paths = layer_paths(layer)
            if not paths["fisher_csv"].exists() or not paths["fisher_metadata"].exists():
                raise FileNotFoundError(f"Layer {layer} has no complete Fisher results")
            fisher_metadata = json.loads(paths["fisher_metadata"].read_text())
            if not fisher_metadata.get("complete"):
                raise RuntimeError(f"Layer {layer} Fisher results are incomplete")
            fisher_hash = file_sha256(paths["fisher_csv"])
            payload = {
                "layer": layer, "fisher_sha256": fisher_hash,
                "subjects": config["subjects"], "data_root": config["data_root"],
                "significant_channels_path": config["significant_channels_path"],
                "participant_map_path": config["participant_map_path"],
                "gpt2_transcript_path": config["gpt2_transcript_path"],
                "lag_tmin": config["lag_tmin"], "lag_tmax": config["lag_tmax"],
                "regression_type": regression_type, "num_folds": num_folds,
                "feature_sets": feature_sets,
            }
            signature, signature_payload = make_signature("feature_models", payload)
            complete = False
            if not refresh and paths["models_npz"].exists() and paths["models_metadata"].exists():
                try:
                    metadata = json.loads(paths["models_metadata"].read_text())
                    with np.load(paths["models_npz"], allow_pickle=False) as saved:
                        correlations = saved["correlations"]
                        complete = (
                            metadata.get("complete") is True
                            and metadata.get("signature") == signature
                            and correlations.ndim == 4
                            and correlations.shape[:2] == (len(feature_sets), num_folds)
                            and correlations.shape[2] == len(saved["electrode_names"])
                            and correlations.shape[2] == len(saved["electrode_subjects"])
                            and correlations.shape[3] == len(saved["lags"])
                            and np.array_equal(saved["model_names"], list(feature_sets))
                            and metadata.get("shape") == list(correlations.shape)
                        )
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    complete = False
            if complete:
                statuses[layer] = "skipped (complete)"
                continue

            features = pd.read_csv(paths["fisher_csv"]).sort_values("word_idx").set_index("word_idx", drop=False)
            unique_words = features["word"].astype(str).unique()
            features["word_frequency"] = features["word"].astype(str).map({
                word: word_frequency(word, "en", minimum=1e-9) for word in unique_words
            })
            features["word_zipf_frequency"] = features["word"].astype(str).map({
                word: zipf_frequency(word, "en", minimum=1.0) for word in unique_words
            })
            subject_results = {name: [] for name in feature_sets}
            electrode_subjects = []
            electrode_names = []
            subject_word_counts = {}
            skipped_channels = 0
            lags = None
            for subject in config["subjects"]:
                if subject not in significant:
                    continue
                cache = load_precomputed_epochs(subject, config["data_root"])
                usable = [ch for ch in significant[subject] if ch in cache["ch_names"]]
                skipped_channels += len(significant[subject]) - len(usable)
                if not usable:
                    continue
                selection = cache["selection"].astype(int)
                sf = features.loc[selection, ["pr", "log_trace", "word_frequency", "word_zipf_frequency"]].copy()
                sf["surprisal"] = word_surprisal.loc[selection].to_numpy()
                sf["entropy"] = word_entropy.loc[selection].to_numpy()
                sf["pr_x_log_trace"] = (
                    (sf["pr"] - sf["pr"].mean())
                    * (sf["log_trace"] - sf["log_trace"].mean())
                )
                if not np.isfinite(sf.to_numpy()).all():
                    raise ValueError(f"Subject {subject} has non-finite model features")
                neural = epochs_from_cache(cache, elec_names=usable)
                subject_lags = np.linspace(
                    config["lag_tmin"], config["lag_tmax"], cache["n_lags"],
                    endpoint=False,
                )
                if lags is None:
                    lags = subject_lags
                elif not np.allclose(lags, subject_lags):
                    raise ValueError("Subjects have inconsistent lag arrays")
                targets = neural.reshape(len(selection), -1)
                epochs_shape = (len(usable), cache["n_lags"])
                for model_name, columns in feature_sets.items():
                    corrs, _, _, _, _ = train_encoding(
                        sf[columns].to_numpy(dtype=np.float64), targets,
                        model_constructor, epochs_shape=epochs_shape,
                        num_folds=num_folds,
                    )
                    subject_results[model_name].append(corrs)
                electrode_subjects.extend([subject] * len(usable))
                electrode_names.extend(map(str, usable))
                subject_word_counts[str(subject)] = len(selection)
            if lags is None:
                raise ValueError("No subjects with usable significant electrodes")
            pooled = {
                name: np.concatenate(values, axis=1)
                for name, values in subject_results.items()
            }
            correlations = np.stack([pooled[name] for name in feature_sets])
            expected = (len(feature_sets), num_folds, len(electrode_names), len(lags))
            if correlations.shape != expected:
                raise AssertionError(f"Unexpected feature-model shape {correlations.shape}")
            paths["models_dir"].mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                paths["models_npz"], correlations=correlations,
                model_names=np.asarray(list(feature_sets)),
                electrode_subjects=np.asarray(electrode_subjects),
                electrode_names=np.asarray(electrode_names), lags=lags,
            )
            summary_rows = []
            for model_name, corrs in pooled.items():
                fold_means = np.nanmean(corrs, axis=1)
                means = np.nanmean(fold_means, axis=0)
                sems = np.nanstd(fold_means, axis=0) / np.sqrt(num_folds)
                for lag, mean, sem in zip(lags, means, sems):
                    summary_rows.append({
                        "model": model_name, "lag": float(lag),
                        "mean_correlation": float(mean),
                        "sem_across_folds": float(sem),
                        "n_electrodes": len(electrode_names), "n_folds": num_folds,
                    })
            pd.DataFrame(summary_rows).to_csv(paths["models_dir"] / "summary.csv", index=False)
            metadata = {
                "complete": True, "signature": signature,
                "analysis_config": signature_payload,
                "shape": list(correlations.shape),
                "axes": ["model", "fold", "electrode", "lag"],
                "subject_word_counts": subject_word_counts,
                "skipped_channels": skipped_channels,
            }
            paths["models_metadata"].write_text(json.dumps(metadata, indent=2))
            statuses[layer] = "computed"
        return statuses

    return (run_feature_models_batch,)


@app.cell(hide_code=True)
def _(
    correlation_button,
    generation_button,
    mo,
    model_fitting_button,
    refresh,
    run_all_button,
    run_ecog_batch,
    run_feature_models_batch,
    run_fisher_batch,
):
    _requested = any(
        button.value for button in (
            generation_button, correlation_button, model_fitting_button, run_all_button
        )
    )
    mo.stop(not _requested, mo.md("Choose an analysis to run."))
    _reports = []
    if run_all_button.value or generation_button.value:
        _reports.append(("Fisher features", run_fisher_batch(refresh=refresh.value)))
    if run_all_button.value or correlation_button.value:
        _reports.append(("ECoG correlations", run_ecog_batch(refresh=refresh.value)))
    if run_all_button.value or model_fitting_button.value:
        _reports.append(("Feature models", run_feature_models_batch(refresh=refresh.value)))
    _lines = ["## Batch results"]
    for _stage, _statuses in _reports:
        _lines.append(f"\n### {_stage}")
        _lines.extend(f"- Layer {_layer}: {_status}" for _layer, _status in _statuses.items())
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Saved-output visualization

    These plots load completed ECoG correlation and feature-model outputs from
    disk. They do not run or refresh any analysis stage.
    """)
    return


@app.cell(hide_code=True)
def _(configured_layers, json, layer_paths, mo, np):
    def load_saved_fisher_analysis(layer, analysis_kind):
        paths = layer_paths(layer)
        if analysis_kind == "ecog":
            data_path = paths["ecog_npz"]
            metadata_path = paths["ecog_metadata"]
            name_key = "feature_names"
        elif analysis_kind == "models":
            data_path = paths["models_npz"]
            metadata_path = paths["models_metadata"]
            name_key = "model_names"
        else:
            raise ValueError(f"Unsupported analysis kind: {analysis_kind}")

        if not data_path.exists() or not metadata_path.exists():
            return None

        try:
            metadata = json.loads(metadata_path.read_text())
            if not metadata.get("complete"):
                return None
            with np.load(data_path, allow_pickle=False) as saved:
                return {
                    "layer": layer,
                    "correlations": saved["correlations"],
                    "names": [str(name) for name in saved[name_key]],
                    "lags": saved["lags"],
                    "electrode_names": [str(name) for name in saved["electrode_names"]],
                    "electrode_subjects": saved["electrode_subjects"],
                }
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def load_saved_fisher_analyses(analysis_kind):
        loaded = {}
        missing_layers = []
        for layer in configured_layers():
            result = load_saved_fisher_analysis(layer, analysis_kind)
            if result is None:
                missing_layers.append(layer)
            else:
                loaded[layer] = result
        return loaded, missing_layers

    def ordered_saved_names(loaded):
        names = []
        for result in loaded.values():
            for name in result["names"]:
                if name not in names:
                    names.append(name)
        return names

    viz_analysis = mo.ui.dropdown(
        options={"ECoG correlations": "ecog", "Feature models": "models"},
        value="ECoG correlations",
        label="Analysis",
    )
    viz_summary = mo.ui.dropdown(
        options={"Mean +/- SEM": "mean_sem", "Median + IQR": "median_iqr"},
        value="Mean +/- SEM",
        label="Summary",
    )
    viz_share_y = mo.ui.checkbox(value=True, label="Shared y-axis")
    mo.hstack([viz_analysis, viz_summary, viz_share_y])
    return (
        load_saved_fisher_analyses,
        ordered_saved_names,
        viz_analysis,
        viz_share_y,
        viz_summary,
    )


@app.cell(hide_code=True)
def _(load_saved_fisher_analyses, mo, ordered_saved_names, viz_analysis):
    _loaded, _missing_layers = load_saved_fisher_analyses(viz_analysis.value)
    _names = ordered_saved_names(_loaded)
    if viz_analysis.value == "ecog":
        _preferred = ["pr", "log_trace", "surprisal"]
    else:
        _preferred = [
            "log word frequency + surprisal + entropy",
            "PR",
            "log trace",
            "PR + log trace + interaction",
        ]
    _default_names = [name for name in _preferred if name in _names]
    if not _default_names:
        _default_names = _names[: min(3, len(_names))]
    viz_selected_names = mo.ui.multiselect(
        options=_names,
        value=_default_names,
        label="Features/models",
    )
    _status = (
        "No completed saved outputs found for this analysis."
        if not _loaded
        else f"Loaded layers: {', '.join(map(str, _loaded))}"
    )
    if _missing_layers:
        _status += f"; missing layers: {', '.join(map(str, _missing_layers))}"
    mo.vstack([viz_selected_names, mo.md(_status)])
    return (viz_selected_names,)


@app.cell(hide_code=True)
def _(
    load_saved_fisher_analyses,
    mo,
    np,
    plt,
    viz_analysis,
    viz_selected_names,
    viz_share_y,
    viz_summary,
):
    def _finite_count(data, axis):
        return np.sum(np.isfinite(data), axis=axis)

    def _mean_sem_band(data, axis):
        mean = np.nanmean(data, axis=axis)
        count = _finite_count(data, axis)
        sem = np.nanstd(data, axis=axis) / np.sqrt(np.maximum(count, 1))
        return mean, mean - sem, mean + sem

    def _median_iqr_band(data, axis):
        return (
            np.nanmedian(data, axis=axis),
            np.nanquantile(data, 0.25, axis=axis),
            np.nanquantile(data, 0.75, axis=axis),
        )

    def _summarize_result(correlations, item_idx, analysis_kind, summary_kind):
        if analysis_kind == "ecog":
            data = correlations[item_idx]
            axis = 0
        else:
            data = np.nanmean(correlations[item_idx], axis=1)
            axis = 0
        if summary_kind == "median_iqr":
            return _median_iqr_band(data, axis=axis)
        return _mean_sem_band(data, axis=axis)

    _loaded, _missing_layers = load_saved_fisher_analyses(viz_analysis.value)
    _selected_names = list(viz_selected_names.value or [])
    mo.stop(not _loaded, mo.md("No completed saved outputs found for this analysis."))
    mo.stop(not _selected_names, mo.md("Select at least one feature/model to display."))

    _layers = list(_loaded)
    _n_layers = len(_layers)
    _n_cols = min(3, _n_layers)
    _n_rows = int(np.ceil(_n_layers / _n_cols))
    _fig, _axes = plt.subplots(
        _n_rows,
        _n_cols,
        figsize=(5.5 * _n_cols, 3.4 * _n_rows),
        sharex=True,
        sharey=viz_share_y.value,
        squeeze=False,
    )
    _colors = plt.get_cmap("tab10").colors

    for _plot_idx, _layer in enumerate(_layers):
        _ax = _axes.flat[_plot_idx]
        _result = _loaded[_layer]
        _name_to_idx = {name: idx for idx, name in enumerate(_result["names"])}
        for _line_idx, _name in enumerate(_selected_names):
            if _name not in _name_to_idx:
                continue
            _center, _lower, _upper = _summarize_result(
                _result["correlations"],
                _name_to_idx[_name],
                viz_analysis.value,
                viz_summary.value,
            )
            _color = _colors[_line_idx % len(_colors)]
            _ax.plot(_result["lags"], _center, color=_color, linewidth=1.5, label=_name)
            _ax.fill_between(_result["lags"], _lower, _upper, color=_color, alpha=0.18)
        _ax.axhline(0, color="black", linewidth=0.6)
        _ax.set_title(f"Layer {_layer}")
        _ax.set_xlabel("Lag (s)")
        _ax.set_ylabel("Correlation")

    for _empty_idx in range(_n_layers, _n_rows * _n_cols):
        _axes.flat[_empty_idx].set_visible(False)

    _handles = []
    _labels = []
    for _ax in _axes.flat:
        _ax_handles, _ax_labels = _ax.get_legend_handles_labels()
        for _handle, _label in zip(_ax_handles, _ax_labels):
            if _label not in _labels:
                _handles.append(_handle)
                _labels.append(_label)
    if _handles:
        _fig.legend(
            _handles,
            _labels,
            loc="upper center",
            ncol=min(3, len(_labels)),
            fontsize=8,
            frameon=False,
        )
    _title = "ECoG correlations" if viz_analysis.value == "ecog" else "Feature models"
    _fig.suptitle(f"{_title}: saved-output lag profiles", y=1.02)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
