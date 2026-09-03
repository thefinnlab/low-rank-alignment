.ONESHELL:

USR := $(shell whoami | head -c 2)
DT  := $(shell date +"%Y%m%d-%H:%M:%S")

# Shared controls. Override any of these on the make command line.
DATASET ?= podcast
MODELS ?= gpt2 gpt2-xl llama3-8b gpt-j-6b roberta-large glove
RRR_MODE ?= cross_subject
CROSS_MODE ?= region
SOURCE_DATASET ?=
CMD ?= sbatch
SBATCH ?= sbatch
OVERRIDES ?=

export PYTHONPATH := $(CURDIR)

# CMD=sbatch is the default. Use CMD="uv run python" for a local run.
define launch
if [ "$(CMD)" = "sbatch" ]; then \
	$(SBATCH) --job-name="$(1)-$$model-$(USR)-$(DT)" submit.sh \
		$(2) --config "$$config" $(OVERRIDES); \
else \
	$(CMD) $(2) --config "$$config" $(OVERRIDES); \
fi
endef

## Run significance once per selected model.
## DATASET: podcast, cars-2, megamind, or fmri
## Example: make significance DATASET=podcast MODELS="gpt2 glove"
significance:
	@set -eu
	mkdir -p logs
	case "$(DATASET)" in
		podcast) config_root="configs/significance"; suffix="/significance.yml" ;;
		cars-2) config_root="configs/brain_treebank/cars_2/significance"; suffix="/significance.yml" ;;
		megamind) config_root="configs/brain_treebank/megamind/significance"; suffix="/significance.yml" ;;
		fmri) config_root="configs/cross_dataset/fmri/significance"; suffix=".yml" ;;
		*) echo "Unsupported significance DATASET '$(DATASET)'" >&2; exit 2 ;;
	esac
	models="$(MODELS)"
	if [ "$(DATASET)" = "fmri" ]; then
		case " $$models " in *" glove "*) echo "Skipping glove: fMRI GloVe is not currently supported" >&2 ;; esac
		models="$(filter-out glove,$(MODELS))"
	fi
	[ -n "$$models" ] || { echo "MODELS contains no models supported by $(DATASET) significance" >&2; exit 2; }
	for model in $$models; do
		config="$$config_root/$$model$$suffix"
		[ -f "$$config" ] || { echo "Missing significance config: $$config" >&2; exit 2; }
	done
	for model in $$models; do
		config="$$config_root/$$model$$suffix"
		echo "Running significance for $(DATASET) / $$model: $$config"
		$(call launch,significance-$(DATASET),scripts/run_significance.py)
	done

## Run one RRR mode once per selected model.
## iEEG RRR_MODE: cross_subject, cross_model_layer, leave_one_subject_out,
##                temporal_windows, cross_temporal_windows, or cross_dataset
## fMRI RRR_MODE: cross_subject (within), cross_model_layer, or cross_dataset
## cross_dataset also requires SOURCE_DATASET.
## Example: make rrr DATASET=fmri RRR_MODE=cross_subject MODELS="gpt2 gpt2-xl"
rrr:
	@set -eu
	mkdir -p logs
	mode="$(RRR_MODE)"
	[ "$$mode" != "internal" ] || mode="cross_subject"
	case "$(DATASET)" in
		podcast) config_root="configs/rrr" ;;
		cars-2) config_root="configs/brain_treebank/cars_2/rrr" ;;
		megamind) config_root="configs/brain_treebank/megamind/rrr" ;;
		fmri)
			case "$$mode" in
				cross_subject) config_root="configs/cross_dataset/fmri/within" ;;
				cross_model_layer) config_root="configs/fmri/within_fmri_cross_model_layer" ;;
				cross_dataset)
					[ -n "$(SOURCE_DATASET)" ] || { echo "SOURCE_DATASET is required for cross_dataset RRR" >&2; exit 2; }
					config_root="configs/cross_dataset/fmri/from_$(SOURCE_DATASET)"
					;;
				*) echo "RRR_MODE '$$mode' is unsupported for fmri" >&2; exit 2 ;;
			esac
			;;
		*) echo "Unsupported RRR DATASET '$(DATASET)'" >&2; exit 2 ;;
	esac
	if [ "$$mode" = "cross_dataset" ] && [ "$(DATASET)" != "fmri" ]; then
		[ -n "$(SOURCE_DATASET)" ] || { echo "SOURCE_DATASET is required for cross_dataset RRR" >&2; exit 2; }
		case "$(DATASET)" in
			podcast|cars-2) config_root="configs/cross_dataset/$(DATASET)/from_$(SOURCE_DATASET)" ;;
			*) echo "cross_dataset RRR is not configured for $(DATASET)" >&2; exit 2 ;;
		esac
	fi
	models="$(MODELS)"
	if [ "$(DATASET)" = "fmri" ]; then
		case " $$models " in *" glove "*) echo "Skipping glove: fMRI GloVe is not currently supported" >&2 ;; esac
		models="$(filter-out glove,$(MODELS))"
	fi
	[ -n "$$models" ] || { echo "MODELS contains no models supported by $(DATASET) RRR" >&2; exit 2; }
	for model in $$models; do
		if [ "$(DATASET)" = "fmri" ] || [ "$$mode" = "cross_dataset" ]; then
			config="$$config_root/$$model.yml"
		else
			config="$$config_root/$$model/$$mode.yml"
		fi
		[ -f "$$config" ] || { echo "Missing RRR config: $$config" >&2; exit 2; }
	done
	for model in $$models; do
		if [ "$(DATASET)" = "fmri" ] || [ "$$mode" = "cross_dataset" ]; then
			config="$$config_root/$$model.yml"
		else
			config="$$config_root/$$model/$$mode.yml"
		fi
		echo "Running $$mode RRR for $(DATASET) / $$model: $$config"
		$(call launch,rrr-$(DATASET)-$$mode,scripts/run_rrr.py)
	done

## Run directed cross-condition comparisons once per selected model.
## Valid pairs: DATASET=podcast CROSS_MODE=region, or
##              DATASET=brain-treebank CROSS_MODE=stimulus
## Example: make cross-condition DATASET=podcast CROSS_MODE=region
cross-condition:
	@set -eu
	mkdir -p logs
	case "$(DATASET):$(CROSS_MODE)" in
		podcast:region|brain-treebank:stimulus|brain_treebank:stimulus) ;;
		fmri:*) echo "Cross-condition comparisons are not currently supported for fmri" >&2; exit 2 ;;
		*) echo "Unsupported cross-condition combination DATASET='$(DATASET)' CROSS_MODE='$(CROSS_MODE)'" >&2; exit 2 ;;
	esac
	[ -n "$(strip $(MODELS))" ] || { echo "MODELS must contain at least one model" >&2; exit 2; }
	config_root="configs/cross_condition/$(CROSS_MODE)"
	for model in $(MODELS); do
		config="$$config_root/$$model.yml"
		[ -f "$$config" ] || { echo "Missing cross-condition config: $$config" >&2; exit 2; }
	done
	for model in $(MODELS); do
		config="$$config_root/$$model.yml"
		echo "Running $(CROSS_MODE) comparison for $(DATASET) / $$model: $$config"
		$(call launch,cross-condition-$(CROSS_MODE),scripts/run_cross_condition_rrr.py)
	done

.PHONY: significance rrr cross-condition
