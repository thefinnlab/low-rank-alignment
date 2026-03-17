.ONESHELL:

USR := $(shell whoami | head -c 2)
DT  := $(shell date +"%Y%m%d-%H:%M:%S")

PREFIX   = significance
JOB_NAME ?= "$(PREFIX)-$(USR)-$(DT)"

# Flip to `python` to run locally instead of submitting to Slurm
CMD = sbatch --job-name=$(JOB_NAME) submit.sh

# ── Configurable defaults (override on the command line) ──────────────────────
DATA_ROOT      ?= data/
OUTPUT_DIR     ?= results/
LAYER          ?= 24
N_PERMUTATIONS ?= 5000
FDR_Q          ?= 0.01
SUBJECTS       ?=

# ── Helpers ───────────────────────────────────────────────────────────────────
# Build the --subjects flag only when SUBJECTS is non-empty
ifdef SUBJECTS
  SUBJECTS_FLAG = --subjects $(SUBJECTS)
else
  SUBJECTS_FLAG =
endif

# ── Targets ───────────────────────────────────────────────────────────────────

## Submit (or run) the full significance pipeline
## Usage:
##   make sig                                         # all subjects, default params
##   make sig SUBJECTS="1 3 5"                        # specific subjects
##   make sig N_PERMUTATIONS=100 OUTPUT_DIR=results_test/  # quick test run
##   make sig CMD=python                              # run locally
sig:
	mkdir -p logs $(OUTPUT_DIR)
	$(CMD) run_significance.py \
		--data-root $(DATA_ROOT) \
		--output-dir $(OUTPUT_DIR) \
		--layer $(LAYER) \
		--n-permutations $(N_PERMUTATIONS) \
		--fdr-q $(FDR_Q) \
		$(SUBJECTS_FLAG)

## Quick smoke-test: 1 subject, 100 permutations, results go to results_test/
sig-test:
	$(MAKE) sig \
		SUBJECTS=1 \
		N_PERMUTATIONS=100 \
		OUTPUT_DIR=results_test/ \
		CMD=python

## Run tests
test:
	python -m pytest tests/ -v

.PHONY: sig sig-test test
