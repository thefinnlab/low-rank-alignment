.ONESHELL:

USR := $(shell whoami | head -c 2)
DT  := $(shell date +"%Y%m%d-%H:%M:%S")

PREFIX   = significance
JOB_NAME ?= "$(PREFIX)-$(USR)-$(DT)"

# Flip to `python` to run locally instead of submitting to Slurm
CMD = sbatch --job-name=$(JOB_NAME) submit.sh
# CMD = python
SBATCH ?= sbatch

export PYTHONPATH := $(CURDIR)

# ── Config defaults (override on the command line) ────────────────────────────
ENCODING_CONFIG ?= configs/encoding/encoding_mid.yml
SIG_CONFIG      ?= configs/significance/significance_default.yml
RRR_CONFIG      ?= configs/rrr/rrr_default.yml
LANGUAGE_DECODING_CONFIG ?= configs/rrr/language_decoding.yml
LANGUAGE_FEATURE_CONFIG ?= configs/language_features/podcast.yml
GCCA_CONFIG     ?= configs/gcca/maxvar_gcca.yml
GCCA_ENCODING_CONFIG ?= configs/gcca/encoding.yml
PRECOMPUTE_EPOCHS_CONFIG ?= configs/precompute_epochs/precompute_epochs_default.yml
EMBED_CONFIG    ?= configs/embeddings/generate_gpt2_xl.yml
FMRI_EMBED_CONFIG ?= configs/fmri/generate_gpt2_xl.yml
FMRI_SIG_CONFIG ?= configs/fmri/significance_gpt2_xl_layer24.yml
FMRI_RRR_CONFIG ?= configs/fmri/cross_dataset_rrr_gpt2_xl_layer24.yml
RANDOM_EMBED_CONFIG ?= configs/embeddings/generate_random_gpt2_xl.yml
PIPELINE_MODELS ?= gpt-j-6b gpt2-xl llama3-8b roberta-large random-gpt2-xl gpt2 glove
DATASET         ?= podcast
SIG_OVERRIDES   ?=
RRR_OVERRIDES   ?=
RUN_DIR         ?=   # required for plots: RUN_DIR=results/my_run
DATA_ROOT       ?= data
ATLAS_PATH      ?= atlases/destrieux_2009/destrieux2009_rois_lateralized.nii.gz
OVERRIDES       ?=   # e.g. OVERRIDES="--encoding.layer=48 --n_permutations=100"

# For downloading model weights.
# MODELS ?= openai-community/gpt2-xl meta-llama/Meta-Llama-3-8B openai-community/gpt2 EleutherAI/gpt-j-6b FacebookAI/roberta-large
# MODELS ?= meta-llama/Llama-3.1-8B

# Batch config lists. Override these with whitespace-separated
# config paths, e.g. SIG_CONFIGS="configs/a.yml configs/b.yml".
EMBED_CONFIGS   ?= configs/embeddings/generate_gpt2_xl.yml \
				   configs/embeddings/generate_gptj_6b.yml \
				   configs/embeddings/generate_random_gpt2_xl.yml \
				   configs/embeddings/generate_llama3_8b.yml \
				   configs/embeddings/generate_roberta_large.yml \
				   configs/embeddings/generate_gpt2.yml
SIG_CONFIGS     ?= configs/significance/gpt-j-6b/significance.yml \
				   configs/significance/gpt2-xl/significance.yml \
				   configs/significance/llama3-8b/significance.yml \
				   configs/significance/roberta-large/significance.yml \
				   configs/significance/random-gpt2-xl/significance.yml \
				   configs/significance/gpt2/significance.yml \
				   configs/significance/glove/significance.yml
RRR_CONFIGS     ?= configs/rrr/gpt-j-6b/cross_subject.yml \
				   configs/rrr/gpt-j-6b/leave_one_subject_out.yml \
				   configs/rrr/gpt2/cross_subject.yml \
				   configs/rrr/gpt2/leave_one_subject_out.yml \
				   configs/rrr/gpt2-xl/cross_subject.yml \
				   configs/rrr/gpt2-xl/leave_one_subject_out.yml \
				   configs/rrr/llama3-8b/cross_subject.yml \
				   configs/rrr/llama3-8b/leave_one_subject_out.yml \
				   configs/rrr/roberta-large/cross_subject.yml \
				   configs/rrr/roberta-large/leave_one_subject_out.yml \
				   configs/rrr/glove/cross_subject.yml \
				   configs/rrr/glove/leave_one_subject_out.yml


define run_config_list
set -e
mkdir -p logs
for config in $($(1)); do
	echo "==> $(2): $$config"
	$(CMD) $(3) --config $$config $(OVERRIDES)
done
endef

download-models:
	@for model in $(MODELS); do \
		echo "Processing $$model..."; \
		SHORT_NAME=$$(echo $$model | cut -d'/' -f2); \
		python scripts/utilities/download_model.py \
			--model-id=$$model \
			--output-dir=model_cache/models/$$SHORT_NAME \
			--token $(TOKEN) \
			--include-tokenizer; \
	done

# ── Targets ───────────────────────────────────────────────────────────────────

## Submit (or run) the full significance pipeline
## Usage:
##   make sig                                                      # default config
##   make sig SIG_CONFIG=configs/significance_fast.yml            # fast config
##   make sig OVERRIDES="--encoding.subjects=[1,3,5]"             # specific subjects
##   make sig CMD=python                                           # run locally
sig:
	mkdir -p logs
	$(CMD) scripts/sig_lags/run_significance.py --config $(SIG_CONFIG) $(OVERRIDES)

## Release significance runs for a list of configs
## Usage:
##   make sig-batch SIG_CONFIGS="configs/significance/a.yml configs/significance/b.yml"
##   make sig-batch SIG_CONFIGS="configs/significance/a.yml configs/significance/b.yml" CMD=python
sig-batch:
	$(call run_config_list,SIG_CONFIGS,significance,scripts/sig_lags/run_significance.py)

## Train encoding models
## Usage:
##   make train                                                    # default config
##   make train ENCODING_CONFIG=configs/encoding_final.yml        # final layer
##   make train OVERRIDES="--run_name=my_run"                     # set run name
##   make train CMD=python                                         # run locally
train:
	mkdir -p logs
	$(CMD) scripts/encoding/train_encoding_models.py --config $(ENCODING_CONFIG) $(OVERRIDES)

## Fit partial-support MAXVAR GCCA over configured saved subspace views.
## Use CMD="uv run python" for a local run.
gcca:
	mkdir -p logs
	$(CMD) scripts/gcca/run_gcca.py --config $(GCCA_CONFIG) $(OVERRIDES)

## Evaluate saved GCCA scores and compare matched targets with existing RRR results.
gcca-encoding:
	mkdir -p logs
	$(CMD) scripts/gcca/run_encoding.py --config $(GCCA_ENCODING_CONFIG) $(OVERRIDES)

## Build dataset-scoped epoch caches used by downstream analysis.
## Usage:
##   make precompute-epochs                                                 # default config
##   make precompute-epochs PRECOMPUTE_EPOCHS_CONFIG=configs/precompute_epochs/precompute_epochs_default.yml  # explicit config
##   make precompute-epochs CMD=python                                      # run locally
precompute-epochs:
	mkdir -p logs
	$(CMD) scripts/utilities/precompute_epochs.py --config $(PRECOMPUTE_EPOCHS_CONFIG) $(OVERRIDES)

## Generate per-token contextual embeddings for the podcast transcript from a HuggingFace model
## Usage:
##   make generate-embeddings                                                 # default: regenerate gpt2-xl
##   make generate-embeddings EMBED_CONFIG=configs/embeddings/generate_gpt2.yml
##   make generate-embeddings EMBED_CONFIG=configs/embeddings/generate_roberta_base.yml
##   make generate-embeddings EMBED_CONFIG="configs/embeddings/generate_gpt2.yml configs/embeddings/generate_gpt2_medium.yml configs/embeddings/generate_gpt2_large.yml configs/embeddings/generate_gpt2_xl.yml"
##                                                                            # run multiple configs sequentially in one job
##   make generate-embeddings OVERRIDES="--context_window=2048 --stride=1024" # overrides apply to every config
##   make generate-embeddings CMD=python                                      # run locally
generate-embeddings:
	mkdir -p logs
	$(CMD) scripts/utilities/generate_embeddings.py $(foreach c,$(EMBED_CONFIG),--config $(c)) $(OVERRIDES)

## Generate contextual embeddings independently for each configured fMRI story.
fmri-generate-embeddings:
	mkdir -p logs
	$(CMD) scripts/fmri/generate_embeddings.py --config $(FMRI_EMBED_CONFIG) $(OVERRIDES)

## Find significant fMRI voxels from session-held-out full-embedding models.
fmri-sig:
	mkdir -p logs
	$(CMD) scripts/fmri/run_significance.py --config $(FMRI_SIG_CONFIG) $(OVERRIDES)

## Evaluate cross-dataset factors or learn fMRI FIR-RRR/PCA axes on significant voxels.
fmri-rrr:
	mkdir -p logs
	$(CMD) scripts/fmri/run_cross_dataset_rrr.py --config $(FMRI_RRR_CONFIG) $(OVERRIDES)

## Release embedding-generation runs for a list of configs
## Usage:
##   make generate-embeddings-batch EMBED_CONFIGS="configs/embeddings/generate_gpt2.yml configs/embeddings/generate_gpt2_xl.yml"
##   make generate-embeddings-batch EMBED_CONFIGS="configs/embeddings/generate_gpt2.yml configs/embeddings/generate_gpt2_xl.yml" CMD=python
generate-embeddings-batch:
	$(call run_config_list,EMBED_CONFIGS,generate embeddings,scripts/utilities/generate_embeddings.py)

## Generate random-init contextual embeddings using the random embedding wrapper
## Usage:
##   make generate-random-embeddings
##   make generate-random-embeddings RANDOM_EMBED_CONFIG=configs/embeddings/generate_random_gpt2_xl.yml CMD=python
generate-random-embeddings:
	mkdir -p logs
	$(CMD) scripts/utilities/generate_random_embeddings.py --config $(RANDOM_EMBED_CONFIG) $(OVERRIDES)

## Generate correlation plots for an existing run directory
## Usage:
##   make plots RUN_DIR=results/my_run                                           # all subjects
##   make plots RUN_DIR=results/my_run OVERRIDES="--subjects 1 3 5"             # specific subjects
##   make plots RUN_DIR=results/my_run OVERRIDES="--force"                      # overwrite existing
##   make plots RUN_DIR=results/my_run OVERRIDES="--threshold 0.90"             # SVD threshold
##   make plots RUN_DIR=results/my_run OVERRIDES="--svd-modes cross_subject"    # specific SVD mode
plots:
	python scripts/utilities/generate_plots.py --run-dir $(RUN_DIR) $(OVERRIDES)

## Run RRR analysis (raw-input baseline: no SVD projection)
## Usage:
##   make rrr                                                          # default config
##   make rrr RRR_CONFIG=configs/rrr/rrr_default.yml                  # per-subject mode
##   make rrr OVERRIDES="--mode=per_region"                           # per-region mode
##   make rrr CMD=python                                               # run locally
rrr:
	mkdir -p logs
	$(CMD) scripts/rrr/run_rrr.py --config $(RRR_CONFIG) $(OVERRIDES)

## Decode linguistic targets from saved RRR input subspaces
## Usage:
##   make language-decoding CMD=python
##   make language-decoding LANGUAGE_DECODING_CONFIG=configs/rrr/language_decoding.yml
language-decoding:
	mkdir -p logs
	$(CMD) scripts/rrr/run_language_decoding.py --config $(LANGUAGE_DECODING_CONFIG) $(OVERRIDES)

## Generate reusable word-level language annotations and structured sidecars
## Usage:
##   make language-features CMD=python
##   make language-features LANGUAGE_FEATURE_CONFIG=configs/language_features/podcast.yml
language-features:
	mkdir -p logs
	$(CMD) scripts/utilities/generate_language_features.py --config $(LANGUAGE_FEATURE_CONFIG) $(OVERRIDES)

## Explicitly download/validate resources without running annotation providers
stage-language-resources:
	$(CMD) scripts/utilities/generate_language_features.py --config $(LANGUAGE_FEATURE_CONFIG) --stage-resources $(OVERRIDES)

## Release RRR runs for a list of configs
## Usage:
##   make rrr-batch RRR_CONFIGS="configs/rrr/rrr_cross_final.yml configs/rrr/rrr_per_subject_final.yml configs/rrr/rrr_per_region_final.yml"
##   make rrr-batch RRR_CONFIGS="configs/rrr/rrr_cross_final.yml configs/rrr/rrr_per_subject_final.yml" CMD=python
rrr-batch:
	$(call run_config_list,RRR_CONFIGS,rrr,scripts/rrr/run_rrr.py)

## Submit paired significance and cross-subject RRR jobs for each model.
## Each RRR job starts only after its model's significance job succeeds.
## Usage:
##   make sig-rrr-batch
##   make sig-rrr-batch PIPELINE_MODELS="gpt2 gpt2-xl" DATASET=podcast
##   make sig-rrr-batch PIPELINE_MODELS="gpt2-xl" DATASET=cars-2 \
##       SIG_OVERRIDES="--n_permutations=100" RRR_OVERRIDES="--refresh=false"
sig-rrr-batch:
	@set -eu; \
	mkdir -p logs; \
	case "$(DATASET)" in \
		podcast) \
			sig_root="configs/significance"; \
			rrr_root="configs/rrr"; \
			;; \
		cars-2) \
			sig_root="configs/brain_treebank/cars_2/significance"; \
			rrr_root="configs/brain_treebank/cars_2/rrr"; \
			;; \
		*) \
			echo "Unsupported DATASET '$(DATASET)'; expected podcast or cars-2" >&2; \
			exit 2; \
			;; \
	esac; \
	if [ -z "$(strip $(PIPELINE_MODELS))" ]; then \
		echo "PIPELINE_MODELS must contain at least one model" >&2; \
		exit 2; \
	fi; \
	for model in $(PIPELINE_MODELS); do \
		sig_config="$$sig_root/$$model/significance.yml"; \
		rrr_config="$$rrr_root/$$model/cross_subject.yml"; \
		if [ ! -f "$$sig_config" ]; then \
			echo "Missing significance config: $$sig_config" >&2; \
			exit 2; \
		fi; \
		if [ ! -f "$$rrr_config" ]; then \
			echo "Missing RRR config: $$rrr_config" >&2; \
			exit 2; \
		fi; \
	done; \
	for model in $(PIPELINE_MODELS); do \
		sig_config="$$sig_root/$$model/significance.yml"; \
		rrr_config="$$rrr_root/$$model/cross_subject.yml"; \
		sig_submission="$$( $(SBATCH) --parsable --job-name="significance-$(USR)-$$model-$(DT)" submit.sh scripts/sig_lags/run_significance.py --config "$$sig_config" $(SIG_OVERRIDES) )"; \
		sig_job_id="$${sig_submission%%;*}"; \
		case "$$sig_job_id" in \
			''|*[!0-9]*) \
				echo "Could not parse significance job ID from: $$sig_submission" >&2; \
				exit 2; \
				;; \
		esac; \
		echo "Submitted significance for $$model: $$sig_job_id"; \
		rrr_submission="$$( $(SBATCH) --parsable --dependency="afterok:$$sig_job_id" --job-name="rrr-cross-subject-$(USR)-$$model-$(DT)" submit.sh scripts/rrr/run_rrr.py --config "$$rrr_config" $(RRR_OVERRIDES) )"; \
		rrr_job_id="$${rrr_submission%%;*}"; \
		case "$$rrr_job_id" in \
			''|*[!0-9]*) \
				echo "Could not parse RRR job ID from: $$rrr_submission" >&2; \
				exit 2; \
				;; \
		esac; \
		echo "Submitted cross-subject RRR for $$model: $$rrr_job_id (afterok:$$sig_job_id)"; \
	done

## Print out regions where electrode in data are located according to an atlas
## Usage:
## make print-regions DATA_ROOT=data RUN_DIR=results/my_run
print-atlas-regions:
	python scripts/utilities/print_atlas_regions.py --data_root=$(DATA_ROOT) --output_dir=$(RUN_DIR) --atlas_path=$(ATLAS_PATH)

## Quick smoke-test: 1 subject, 100 permutations
sig-test:
	$(MAKE) sig SIG_CONFIG=configs/significance_fast.yml OVERRIDES="--encoding.subjects=[1]" CMD=python

## Run tests
test:
	python -m pytest tests/ -v

.PHONY: sig sig-batch train plots rrr rrr-batch sig-rrr-batch sig-test test generate-embeddings generate-embeddings-batch generate-random-embeddings download-models precompute-epochs print-atlas-regions
