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
ENCODING_CONFIG ?= configs/encoding_mid.yml
SIG_CONFIG      ?= configs/significance_default.yml
SVD_CONFIG      ?= configs/svd_default.yml
SRRR_CONFIG     ?= configs/srrr_default.yml
RRR_CONFIG      ?= configs/rrr_default.yml
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
##   make svd SVD_CONFIG=configs/svd_default.yml                      # explicit config
##   make svd OVERRIDES="--per_subject_svd=true"                      # per-subject mode
##   make svd CMD=python                                               # run locally
svd:
	mkdir -p logs
	$(CMD) scripts/run_svd_analysis.py --config $(SVD_CONFIG) $(OVERRIDES)

## Run SRRR analysis to find minimal subset of SVD directions
## Usage:
##   make srrr                                                         # default config
##   make srrr SRRR_CONFIG=configs/srrr_default.yml                   # explicit config
##   make srrr OVERRIDES="--mode=per_subject"                         # per-subject mode
##   make srrr CMD=python                                              # run locally
srrr:
	mkdir -p logs
	$(CMD) scripts/run_srrr.py --config $(SRRR_CONFIG) $(OVERRIDES)

## Run RRR analysis (raw-input baseline: no SVD projection)
## Usage:
##   make rrr                                                          # default config
##   make rrr RRR_CONFIG=configs/rrr_per_subject.yml                  # per-subject mode
##   make rrr OVERRIDES="--mode=per_region"                           # per-region mode
##   make rrr CMD=python                                               # run locally
rrr:
	mkdir -p logs
	$(CMD) scripts/run_rrr.py --config $(RRR_CONFIG) $(OVERRIDES)

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

.PHONY: sig train plots svd srrr rrr partial-svd sig-test test
