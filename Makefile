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
SIG_CONFIG      ?= configs/significance/significance_default.yml
RRR_CONFIG      ?= configs/rrr/rrr_default.yml
LANGUAGE_DECODING_CONFIG ?= configs/rrr/language_decoding.yml
LANGUAGE_RECONSTRUCTION_CONFIG ?= configs/rrr/language_reconstruction.yml
SUBSPACE_COMPARISON_CONFIG ?= configs/rrr/compare_subspaces.yml
SUBSPACE_COMPARISON_CONFIGS ?= configs/rrr/compare_subspaces_podcast.yml \
							  configs/rrr/compare_subspaces_cars-2.yml \
							  configs/rrr/compare_subspaces_fmri.yml
LANGUAGE_FEATURE_CONFIG ?= configs/language_features/podcast.yml
SURPRISAL_CONFIGS ?= configs/language_features/podcast.yml configs/language_features/brain_treebank.yml configs/language_features/fmri.yml
SELF_SURPRISAL_CONFIG ?= configs/language_features/self_surprisal.yml
PRECOMPUTE_EPOCHS_CONFIG ?= configs/precompute_epochs/precompute_epochs_default.yml
EMBED_CONFIG    ?= configs/embeddings/generate_gpt2_xl.yml
FMRI_EMBED_CONFIG ?= configs/fmri/generate_gpt2_xl.yml
FMRI_SIG_CONFIG ?= configs/fmri/significance_gpt2_xl_layer24.yml
FMRI_RRR_CONFIG ?= configs/fmri/cross_dataset_rrr_gpt2_xl_layer24.yml
RANDOM_EMBED_CONFIG ?= configs/embeddings/generate_random_gpt2_xl.yml
PIPELINE_MODELS ?= gpt-j-6b gpt2-xl llama3-8b roberta-large random-gpt2-xl gpt2 glove
CROSS_MODEL_LAYER_MODELS ?= gpt-j-6b gpt2-xl llama3-8b roberta-large random-gpt2-xl gpt2
CROSS_MODEL_LAYER_CARS_MODELS ?= $(CROSS_MODEL_LAYER_MODELS) gpt2-large gpt2-medium pythia-1.4b pythia-160m pythia-6.9b roberta-base
CROSS_MODEL_LAYER_INCLUDE_FMRI ?= true
SIG_RRR_MODELS ?= $(CROSS_MODEL_LAYER_MODELS)
SIG_RRR_CARS_MODELS ?= $(CROSS_MODEL_LAYER_CARS_MODELS)
SIG_RRR_FMRI_MODELS ?= $(CROSS_MODEL_LAYER_MODELS)
DATASET         ?= podcast
SIG_OVERRIDES   ?=
RRR_OVERRIDES   ?=
RUN_DIR         ?=   # required for plots: RUN_DIR=results/my_run
DATA_ROOT       ?= data
ATLAS_PATH      ?= atlases/destrieux_2009/destrieux2009_rois_lateralized.nii.gz
OVERRIDES       ?=   # e.g. OVERRIDES="--encoding.layer=48 --n_permutations=100"
SUBSPACE_COMPARISON_SBATCH_ARGS ?=
TEMPORAL_WINDOW_DATASETS ?= podcast cars-2
TEMPORAL_WINDOW_SBATCH_ARGS ?=

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

# ── Targets ───────────────────────────────────────────────────────────────────

## Submit (or run) the full significance pipeline
## Usage:
##   make sig                                                      # default config
##   make sig SIG_CONFIG=configs/significance_fast.yml            # fast config
##   make sig OVERRIDES="--encoding.subjects=[1,3,5]"             # specific subjects
##   make sig CMD=python                                           # run locally
sig:
	mkdir -p logs
	$(CMD) scripts/run_significance.py --config $(SIG_CONFIG) $(OVERRIDES)

## Release significance runs for a list of configs
## Usage:
##   make sig-batch SIG_CONFIGS="configs/significance/a.yml configs/significance/b.yml"
##   make sig-batch SIG_CONFIGS="configs/significance/a.yml configs/significance/b.yml" CMD=python
sig-batch:
	$(call run_config_list,SIG_CONFIGS,significance,scripts/run_significance.py)

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
	$(CMD) scripts/run_significance.py --config $(FMRI_SIG_CONFIG) $(OVERRIDES)

## Evaluate cross-dataset factors or learn fMRI FIR-RRR/PCA axes on significant voxels.
fmri-rrr:
	mkdir -p logs
	$(CMD) scripts/run_rrr.py --config $(FMRI_RRR_CONFIG) $(OVERRIDES)

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
	$(CMD) scripts/run_rrr.py --config $(RRR_CONFIG) $(OVERRIDES)

## Compare configured saved RRR/PCA spaces over a shared evaluation stimulus.
## Use CMD="uv run python" for a local run.
compare-subspaces:
	mkdir -p logs
	$(CMD) scripts/rrr/compare_subspaces.py --config $(SUBSPACE_COMPARISON_CONFIG) $(OVERRIDES)

## Submit one saved-subspace comparison job per evaluation dataset.
## This target only runs compare_subspaces.py; it never produces missing RRR inputs.
## Usage:
##   make compare-subspaces-batch
##   make compare-subspaces-batch OVERRIDES="--max_rank=16"
##   make compare-subspaces-batch SUBSPACE_COMPARISON_SBATCH_ARGS="--time=08:00:00"
compare-subspaces-batch:
	@set -eu; \
	mkdir -p logs; \
	for config in $(SUBSPACE_COMPARISON_CONFIGS); do \
		dataset="$${config##*compare_subspaces_}"; \
		dataset="$${dataset%.yml}"; \
		if [ ! -f "$$config" ]; then \
			echo "Missing subspace comparison config: $$config" >&2; \
			exit 2; \
		fi; \
		echo "Submitting subspace comparison over $$dataset: $$config"; \
		$(SBATCH) --job-name="subspace-$$dataset-$(USR)-$(DT)" \
			$(SUBSPACE_COMPARISON_SBATCH_ARGS) submit.sh \
			scripts/rrr/compare_subspaces.py --config "$$config" $(OVERRIDES); \
	done

## Decode linguistic targets from saved RRR input subspaces
## Usage:
##   make language-decoding CMD=python
##   make language-decoding LANGUAGE_DECODING_CONFIG=configs/rrr/language_decoding.yml
language-decoding:
	mkdir -p logs
	$(CMD) scripts/language-interpret/run_language_decoding.py --config $(LANGUAGE_DECODING_CONFIG) $(OVERRIDES)

## Predict matched-rank RRR/PCA spaces from configured language-task variables.
## Use CMD="uv run python" for a local run.
language-reconstruction:
	mkdir -p logs
	$(CMD) scripts/language-interpret/run_language_reconstruction.py --config $(LANGUAGE_RECONSTRUCTION_CONFIG) $(OVERRIDES)

## Generate reusable word-level language annotations and structured sidecars
## Usage:
##   make language-features CMD=python
##   make language-features LANGUAGE_FEATURE_CONFIG=configs/language_features/podcast.yml
language-features:
	mkdir -p logs
	$(CMD) scripts/utilities/generate_language_features.py --config $(LANGUAGE_FEATURE_CONFIG) $(OVERRIDES)

## Generate/update only standardized GPT-2 surprisal and entropy across datasets
surprisal:
	$(CMD) scripts/utilities/generate_surprisal.py $(foreach config,$(SURPRISAL_CONFIGS),--config $(config)) $(OVERRIDES)

## Generate model-specific surprisal and entropy used by language decoding
self-surprisal:
	$(CMD) scripts/utilities/generate_self_surprisal.py --config $(SELF_SURPRISAL_CONFIG) $(OVERRIDES)

## Populate fMRI IU position, onset, end, and length from aligned PSST boundaries
fmri-iu-features:
	$(CMD) scripts/utilities/generate_fmri_iu_features.py --config configs/language_features/fmri.yml

## Populate the common interpretable lexical, sentence, norm, and syntax features
fmri-interpretable-features:
	$(CMD) scripts/utilities/generate_fmri_interpretable_features.py --config configs/language_features/fmri.yml

## Populate only Cars 2 concreteness labels without rerunning other providers
cars-2-concreteness:
	$(CMD) scripts/utilities/generate_cars2_concreteness.py --config configs/language_features/brain_treebank.yml

## Resume fMRI SaT/PSST boundaries, then generate expanded Cars 2 and fMRI labels.
expanded-language-features:
	uv run python scripts/utilities/generate_fmri_language_boundaries.py --device cuda --stories alternateithicatom souls avatar legacy odetostepfather undertheinfluence howtodraw myfirstdaywiththeyankees naked stagefright tildeath sloth exorcism haveyoumethimyet adollshouse inamoment theclosetthatateeverything adventuresinsayingyes buck swimmingwithastronauts thatthingonmyarm eyespy itsabox hangtime wheretheressmoke
	uv run python scripts/utilities/generate_language_features.py --config configs/language_features/brain_treebank.yml
	uv run python scripts/utilities/generate_language_features.py --config configs/language_features/fmri.yml

## Explicitly download/validate resources without running annotation providers
stage-language-resources:
	$(CMD) scripts/utilities/generate_language_features.py --config $(LANGUAGE_FEATURE_CONFIG) --stage-resources $(OVERRIDES)

## Release RRR runs for a list of configs
## Usage:
##   make rrr-batch RRR_CONFIGS="configs/rrr/rrr_cross_final.yml configs/rrr/rrr_per_subject_final.yml configs/rrr/rrr_per_region_final.yml"
##   make rrr-batch RRR_CONFIGS="configs/rrr/rrr_cross_final.yml configs/rrr/rrr_per_subject_final.yml" CMD=python
rrr-batch:
	$(call run_config_list,RRR_CONFIGS,rrr,scripts/run_rrr.py)

## Run cross-model-layer significance-union RRR for every non-GloVe model.
## Each model uses its own config, so model-specific dimensions and ridge
## memory settings can be kept alongside the cross-model-layer selection.
## Usage:
##   make rrr-cross-model-layer-batch
##   make rrr-cross-model-layer-batch DATASET=cars-2
##   make rrr-cross-model-layer-batch CMD=python CROSS_MODEL_LAYER_MODELS="gpt2 gpt2-xl"
rrr-cross-model-layer-batch:
	@set -eu; \
	mkdir -p logs; \
	case "$(DATASET)" in \
		podcast) config_root="configs/rrr"; models="$(CROSS_MODEL_LAYER_MODELS)" ;; \
		cars-2) config_root="configs/brain_treebank/cars_2/rrr"; models="$(CROSS_MODEL_LAYER_CARS_MODELS)" ;; \
		*) echo "Unsupported DATASET '$(DATASET)'; expected podcast or cars-2" >&2; exit 2 ;; \
	esac; \
	for model in $$models; do \
		config="$$config_root/$$model/cross_model_layer.yml"; \
		if [ ! -f "$$config" ]; then echo "Missing RRR config: $$config" >&2; exit 2; fi; \
		echo "Submitting cross-model-layer RRR for $(DATASET) / $$model"; \
		$(if $(filter sbatch%,$(CMD)),sbatch --job-name="rrr-$(DATASET)-$$model-$(USR)-$(DT)" submit.sh,$(CMD)) scripts/run_rrr.py --config "$$config" $(RRR_OVERRIDES); \
	done; \
	if [ "$(CROSS_MODEL_LAYER_INCLUDE_FMRI)" = "true" ]; then \
		fmri_config_root="configs/fmri/within_fmri_cross_model_layer"; \
		for model in $(CROSS_MODEL_LAYER_MODELS); do \
			fmri_config="$$fmri_config_root/$$model.yml"; \
			if [ ! -f "$$fmri_config" ]; then echo "Missing fMRI RRR config: $$fmri_config" >&2; exit 2; fi; \
			echo "Submitting cross-model-layer fMRI RRR for $$model"; \
			$(if $(filter sbatch%,$(CMD)),sbatch --job-name="rrr-fmri-$$model-$(USR)-$(DT)" submit.sh,$(CMD)) scripts/run_rrr.py --config "$$fmri_config" $(RRR_OVERRIDES); \
		done; \
	fi

## Submit significance for every union source, then submit dependent union RRR.
## Supports podcast, cars-2, and deep-fMRI.
## Usage:
##   make sig-rrr-cross-model-layer-batch DATASET=podcast
##   make sig-rrr-cross-model-layer-batch DATASET=cars-2
##   make sig-rrr-cross-model-layer-batch DATASET=fmri
##   make sig-rrr-cross-model-layer-batch DATASET=podcast SIG_RRR_MODELS="gpt2 gpt2-xl"
sig-rrr-cross-model-layer-batch:
	@set -eu; \
	mkdir -p logs; \
	case "$(DATASET)" in \
		podcast) \
			sig_root="configs/significance"; \
			rrr_root="configs/rrr"; \
			models="$(SIG_RRR_MODELS)"; \
			sig_override="--ieeg.encoding.shared_word_selection=true"; \
			rrr_modality="ieeg"; \
			;; \
		cars-2) \
			sig_root="configs/brain_treebank/cars_2/significance"; \
			rrr_root="configs/brain_treebank/cars_2/rrr"; \
			models="$(SIG_RRR_CARS_MODELS)"; \
			sig_override=""; \
			rrr_modality="ieeg"; \
			;; \
		fmri) \
			sig_root="configs/cross_dataset/fmri/significance"; \
			rrr_root="configs/fmri/within_fmri_cross_model_layer"; \
			models="$(SIG_RRR_FMRI_MODELS)"; \
			sig_override=""; \
			rrr_modality="fmri"; \
			;; \
		*) echo "Unsupported DATASET '$(DATASET)'; expected podcast, cars-2, or fmri" >&2; exit 2 ;; \
	esac; \
	sig_ids=""; \
	for model in $$models; do \
		if [ "$$rrr_modality" = "fmri" ]; then \
			sig_config="$$sig_root/$$model.yml"; \
		else \
			sig_config="$$sig_root/$$model/significance.yml"; \
		fi; \
		if [ ! -f "$$sig_config" ]; then echo "Missing significance config: $$sig_config" >&2; exit 2; fi; \
		sig_submission="$$( $(SBATCH) --parsable --job-name="sig-$(DATASET)-$$model-$(USR)-$(DT)" submit.sh scripts/run_significance.py --config "$$sig_config" $$sig_override $(SIG_OVERRIDES) )"; \
		sig_id="$${sig_submission%%;*}"; \
		case "$$sig_id" in ''|*[!0-9]*) echo "Could not parse significance job ID: $$sig_submission" >&2; exit 2 ;; esac; \
		sig_ids="$${sig_ids:+$$sig_ids:}$$sig_id"; \
		echo "Submitted significance for $$model: $$sig_id"; \
	done; \
	for model in $$models; do \
		if [ "$$rrr_modality" = "fmri" ]; then \
			rrr_config="$$rrr_root/$$model.yml"; \
		else \
			rrr_config="$$rrr_root/$$model/cross_model_layer.yml"; \
		fi; \
		if [ ! -f "$$rrr_config" ]; then echo "Missing RRR config: $$rrr_config" >&2; exit 2; fi; \
		case "$$model" in \
			gpt-j-6b) model_dir="gpt-j-6b"; layers='[0,7,14,21,28]' ;; \
			gpt2-xl) model_dir="gpt2-xl-batch"; layers='[0,12,24,36,48]' ;; \
			llama3-8b) model_dir="llama3-8b"; layers='[0,8,16,24,32]' ;; \
			roberta-large) model_dir="roberta-large"; layers='[0,6,12,18,24]' ;; \
			random-gpt2-xl) model_dir="random-gpt2-xl"; layers='[0,12,24,36,48]' ;; \
			gpt2) model_dir="gpt2"; layers='[0,3,6,9,12]' ;; \
			gpt2-large) model_dir="gpt2-large"; layers='[0,18,36]' ;; \
			gpt2-medium) model_dir="gpt2-medium"; layers='[0,12,24]' ;; \
			pythia-1.4b) model_dir="pythia-1.4b"; layers='[0,12,24]' ;; \
			pythia-160m) model_dir="pythia-160m"; layers='[0,6,12]' ;; \
			pythia-6.9b) model_dir="pythia-6.9b"; layers='[0,16,32]' ;; \
			roberta-base) model_dir="roberta-base"; layers='[0,6,12]' ;; \
			*) echo "Unsupported union model '$$model'" >&2; exit 2 ;; \
		esac; \
		echo "Submitting dependent union RRR for $(DATASET) / $$model"; \
		if [ "$$rrr_modality" = "fmri" ]; then \
			$(SBATCH) --dependency="afterok:$$sig_ids" --job-name="rrr-fmri-$$model-$(USR)-$(DT)" submit.sh scripts/run_rrr.py --config "$$rrr_config" $(RRR_OVERRIDES); \
		else \
			$(SBATCH) --dependency="afterok:$$sig_ids" --job-name="rrr-$(DATASET)-$$model-$(USR)-$(DT)" submit.sh scripts/run_rrr.py --config "$$rrr_config" --ieeg.encoding.per_model_config.model_dir="$$model_dir" --ieeg.encoding.per_model_config.layer="$$layers" --significance_selection.reference_artifact.model="$$model_dir" $(RRR_OVERRIDES); \
		fi; \
	done

## Submit one temporal-window RRR job per model and dataset. Each config keeps
## all sampled layers in one process, so the default submits 14 jobs rather than
## one job per layer.
## Usage:
##   make temporal-windows-batch
##   make temporal-windows-batch PIPELINE_MODELS="gpt2 gpt2-xl"
##   make temporal-windows-batch TEMPORAL_WINDOW_DATASETS="podcast"
##   make temporal-windows-batch TEMPORAL_WINDOW_SBATCH_ARGS="--time=24:00:00"
temporal-windows-batch:
	@set -eu; \
	mkdir -p logs; \
	if [ -z "$(strip $(PIPELINE_MODELS))" ]; then \
		echo "PIPELINE_MODELS must contain at least one model" >&2; \
		exit 2; \
	fi; \
	for dataset in $(TEMPORAL_WINDOW_DATASETS); do \
		case "$$dataset" in \
			podcast) config_root="configs/rrr" ;; \
			cars-2) config_root="configs/brain_treebank/cars_2/rrr" ;; \
			*) \
				echo "Unsupported temporal-window dataset '$$dataset'; expected podcast or cars-2" >&2; \
				exit 2; \
				;; \
		esac; \
		for model in $(PIPELINE_MODELS); do \
			config="$$config_root/$$model/temporal_windows.yml"; \
			if [ ! -f "$$config" ]; then \
				echo "Missing temporal-window config: $$config" >&2; \
				exit 2; \
			fi; \
		done; \
	done; \
	for dataset in $(TEMPORAL_WINDOW_DATASETS); do \
		case "$$dataset" in \
			podcast) config_root="configs/rrr" ;; \
			cars-2) config_root="configs/brain_treebank/cars_2/rrr" ;; \
		esac; \
		for model in $(PIPELINE_MODELS); do \
			config="$$config_root/$$model/temporal_windows.yml"; \
			echo "Submitting temporal-window RRR for $$dataset / $$model: $$config"; \
			$(SBATCH) --job-name="rrr-windows-$$dataset-$$model-$(USR)-$(DT)" \
				$(TEMPORAL_WINDOW_SBATCH_ARGS) submit.sh \
				scripts/run_rrr.py --config "$$config" $(RRR_OVERRIDES); \
		done; \
	done

## Submit paired significance and internal RRR jobs for each model.
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
		sig_submission="$$( $(SBATCH) --parsable --job-name="significance-$(USR)-$$model-$(DT)" submit.sh scripts/run_significance.py --config "$$sig_config" $(SIG_OVERRIDES) )"; \
		sig_job_id="$${sig_submission%%;*}"; \
		case "$$sig_job_id" in \
			''|*[!0-9]*) \
				echo "Could not parse significance job ID from: $$sig_submission" >&2; \
				exit 2; \
				;; \
		esac; \
		echo "Submitted significance for $$model: $$sig_job_id"; \
		rrr_submission="$$( $(SBATCH) --parsable --dependency="afterok:$$sig_job_id" --job-name="rrr-internal-$(USR)-$$model-$(DT)" submit.sh scripts/run_rrr.py --config "$$rrr_config" $(RRR_OVERRIDES) )"; \
		rrr_job_id="$${rrr_submission%%;*}"; \
		case "$$rrr_job_id" in \
			''|*[!0-9]*) \
				echo "Could not parse RRR job ID from: $$rrr_submission" >&2; \
				exit 2; \
				;; \
		esac; \
		echo "Submitted internal RRR for $$model: $$rrr_job_id (afterok:$$sig_job_id)"; \
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

.PHONY: sig sig-batch train plots rrr rrr-batch rrr-cross-model-layer-batch temporal-windows-batch sig-rrr-batch submit-rrr-matrix sig-test test generate-embeddings generate-embeddings-batch generate-random-embeddings download-models precompute-epochs print-atlas-regions compare-subspaces compare-subspaces-batch
