.ONESHELL:

USR := $(shell whoami | head -c 2)
DT  := $(shell date +"%Y%m%d-%H:%M:%S")

PREFIX   = significance
JOB_NAME ?= "$(PREFIX)-$(USR)-$(DT)"

# Flip to `python` to run locally instead of submitting to Slurm
CMD = sbatch --job-name=$(JOB_NAME) submit.sh
# CMD = python

export PYTHONPATH := $(CURDIR)

# ── Config defaults (override on the command line) ────────────────────────────
ENCODING_CONFIG ?= configs/encoding/encoding_mid.yml
SIG_CONFIG      ?= configs/significance/significance_default.yml
SVD_CONFIG      ?= configs/svd/svd_default.yml
SRRR_CONFIG     ?= configs/srrr/srrr_default.yml
RRR_CONFIG      ?= configs/rrr/rrr_default.yml
PCA_CONFIG      ?= configs/pca/pca_default.yml
ABLATION_CONFIG ?= configs/ablation/ablation_default.yml
NULL_CONFIG	    ?= configs/new_sig/null_dist.yml
NEW_SIG_CONFIG  ?= configs/new_sig/new_sig.yml
PRECOMPUTE_EPOCHS_CONFIG ?= configs/precompute_epochs/precompute_epochs_default.yml
RUN_DIR         ?=   # required for plots: RUN_DIR=results/my_run
DATA_ROOT       ?= data
ATLAS_PATH      ?= atlases/destrieux_2009/destrieux2009_rois_lateralized.nii.gz
OVERRIDES       ?=   # e.g. OVERRIDES="--encoding.layer=48 --n_permutations=100"

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

## Train encoding models
## Usage:
##   make train                                                    # default config
##   make train ENCODING_CONFIG=configs/encoding_final.yml        # final layer
##   make train OVERRIDES="--run_name=my_run"                     # set run name
##   make train CMD=python                                         # run locally
train:
	mkdir -p logs
	$(CMD) scripts/train_encoding_models.py --config $(ENCODING_CONFIG) $(OVERRIDES)

## Test effects of ablating different portions of either the center or tail of RRR axes.
## Usage:
##   make ablation                                                 # default config
##   make ablation ABLATION_CONFIG=configs/ablation/ablation_default.yml  # explicit config
##   make ablation OVERRIDES="--mode=per_subject"                  # per-subject mode
##   make ablation CMD=python                                      # run locally
ablation:
	mkdir -p logs
	$(CMD) scripts/run_ablation.py --config $(ABLATION_CONFIG) $(OVERRIDES)

## Test effects of ablating different portions of either the center or tail of RRR axes.
## Usage:
##   make precompute-epochs                                                 # default config
##   make precompute-epochs PRECOMPUTE_EPOCHS_CONFIG=configs/precompute_epochs/precompute_epochs_default.yml  # explicit config
##   make precompute-epochs CMD=python                                      # run locally
precompute-epochs:
	mkdir -p logs
	$(CMD) scripts/precompute_epochs.py --config $(PRECOMPUTE_EPOCHS_CONFIG) $(OVERRIDES)

## Generate null distribution for new significance test.
## Usage:
##   make null-dist                                                 # default config
##   make null-dist NULL_CONFIG=configs/new_sig/null_dist.yml  # explicit config
##   make null-dist CMD=python                                      # run locally
null-dist:
	mkdir -p logs
	$(CMD) scripts/generate_null_distribution.py --config $(NULL_CONFIG) $(OVERRIDES)

## Apply the new significance test to an existing run directory.
## Usage:
##   make new-sig                                                 # default config
##   make new-sig NEW_SIG_CONFIG=configs/new_sig/new_sig.yml  # explicit config
##   make new-sig CMD=python                                      # run locally
new-sig:
	mkdir -p logs
	$(CMD) scripts/compute_significance.py --config $(NEW_SIG_CONFIG) $(OVERRIDES)

## Generate correlation plots for an existing run directory
## Usage:
##   make plots RUN_DIR=results/my_run                                           # all subjects
##   make plots RUN_DIR=results/my_run OVERRIDES="--subjects 1 3 5"             # specific subjects
##   make plots RUN_DIR=results/my_run OVERRIDES="--force"                      # overwrite existing
##   make plots RUN_DIR=results/my_run OVERRIDES="--threshold 0.90"             # SVD threshold
##   make plots RUN_DIR=results/my_run OVERRIDES="--svd-modes cross_subject"    # specific SVD mode
plots:
	python scripts/generate_plots.py --run-dir $(RUN_DIR) $(OVERRIDES)

## Run SVD analysis on encoding model weights
## Usage:
##   make svd                                                          # default config
##   make svd SVD_CONFIG=configs/svd/svd_default.yml                      # explicit config
##   make svd OVERRIDES="--per_subject_svd=true"                      # per-subject mode
##   make svd CMD=python                                               # run locally
svd:
	mkdir -p logs
	$(CMD) scripts/run_svd_analysis.py --config $(SVD_CONFIG) $(OVERRIDES)

## Run SRRR analysis to find minimal subset of SVD directions
## Usage:
##   make srrr                                                         # default config
##   make srrr SRRR_CONFIG=configs/srrr/srrr_default.yml                   # explicit config
##   make srrr OVERRIDES="--mode=per_subject"                         # per-subject mode
##   make srrr CMD=python                                              # run locally
srrr:
	mkdir -p logs
	$(CMD) scripts/run_srrr.py --config $(SRRR_CONFIG) $(OVERRIDES)

## Run RRR analysis (raw-input baseline: no SVD projection)
## Usage:
##   make rrr                                                          # default config
##   make rrr RRR_CONFIG=configs/rrr/rrr_default.yml                  # per-subject mode
##   make rrr OVERRIDES="--mode=per_region"                           # per-region mode
##   make rrr CMD=python                                               # run locally
rrr:
	mkdir -p logs
	$(CMD) scripts/run_rrr.py --config $(RRR_CONFIG) $(OVERRIDES)

## Run PCA analysis (embedding PCA subspace baseline)
## Usage:
##   make pca                                                          # default config
##   make pca PCA_CONFIG=configs/pca/pca_default.yml                  # explicit config
##   make pca OVERRIDES="--mode=per_region"                           # per-region mode
##   make pca CMD=python                                               # run locally
pca:
	mkdir -p logs
	$(CMD) scripts/run_pca.py --config $(PCA_CONFIG) $(OVERRIDES)

## Run partial SVD analysis (direction importance via partialling)
## Usage:
##   make partial-svd                                                     # default config
##   make partial-svd PARTIAL_SVD_CONFIG=configs/partial_svd_per_region.yml  # per-region plots
##   make partial-svd OVERRIDES="--partial_group_sizes=[1,5,20]"          # custom sweep
##   make partial-svd CMD=python                                          # run locally
PARTIAL_SVD_CONFIG ?= configs/partial_svd_default.yml

partial-svd:
	mkdir -p logs
	$(CMD) scripts/run_partial_svd.py --config $(PARTIAL_SVD_CONFIG) $(OVERRIDES)

## Run partial RRR analysis (subspace importance via partialling)
## Usage:
##   make partial-rrr                                                      # default config
##   make partial-rrr PARTIAL_RRR_CONFIG=configs/partial_rrr_default.yml  # explicit config
##   make partial-rrr OVERRIDES="--mode=per_subject"                      # per-subject mode
##   make partial-rrr CMD=python                                           # run locally
PARTIAL_RRR_CONFIG ?= configs/partial_rrr/partial_rrr_default.yml

partial-rrr:
	mkdir -p logs
	$(CMD) scripts/run_partial_rrr.py --config $(PARTIAL_RRR_CONFIG) $(OVERRIDES)

## Run SAE analysis (sparse autoencoder on RRR-projected embeddings)
## Usage:
##   make sae                                                               # default config
##   make sae SAE_CONFIG=configs/sae/sae_default.yml                           # explicit config
##   make sae OVERRIDES="--k=20 --hidden_dim=100"                          # override params
##   make sae OVERRIDES="--mode=per_subject --max_subjects=3"              # limit subjects
##   make sae CMD=python                                                    # run locally
SAE_CONFIG ?= configs/sae/sae_default.yml

sae:
	mkdir -p logs
	$(CMD) scripts/run_sae.py --config $(SAE_CONFIG) $(OVERRIDES)

## Print out regions where electrode in data are located according to an atlas
## Usage:
## make print-regions DATA_ROOT=data RUN_DIR=results/my_run
print-atlas-regions:
	python scripts/print_atlas_regions.py --data_root=$(DATA_ROOT) --output_dir=$(RUN_DIR) --atlas_path=$(ATLAS_PATH)

## Quick smoke-test: 1 subject, 100 permutations
sig-test:
	$(MAKE) sig SIG_CONFIG=configs/significance_fast.yml OVERRIDES="--encoding.subjects=[1]" CMD=python

## Run tests
test:
	python -m pytest tests/ -v

.PHONY: sig train plots svd srrr rrr pca partial-svd partial-rrr sae sig-test test
