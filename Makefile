.ONESHELL:

USR := $(shell whoami | head -c 2)
DT  := $(shell date +"%Y%m%d-%H:%M:%S")

# Shared controls. Override any of these on the make command line.
DATASET ?= podcast
MODELS ?= gpt2 gpt2-xl llama3-8b gpt-j-6b roberta-large random-gpt2 glove
FINAL_MODELS ?= gpt2 gpt2-xl gpt-j-6b roberta-large llama3-8b random-gpt2 glove
SIGNIFICANCE_MODELS := gpt2 gpt2-xl gpt-j-6b roberta-large llama3-8b random-gpt2 glove
RRR_MODE ?= cross_subject
CROSS_MODE ?= region
SOURCE_DATASET ?=
CMD ?= sbatch
SBATCH ?= sbatch
OVERRIDES ?=
SIGNIFICANCE_CHECK ?= python -m scripts.utilities.check_final_significance

export PYTHONPATH := $(CURDIR)

# CMD=sbatch is the default. Use CMD="uv run python" for a local run.
define launch
if [ "$(CMD)" = "sbatch" ]; then \
	$(SBATCH) --job-name="$(1)-$$model-$(USR)-$(DT)" submit.sh \
		$(2) --config "$$config" $(OVERRIDES) $(3); \
else \
	$(CMD) $(2) --config "$$config" $(OVERRIDES) $(3); \
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

## Submit significance jobs only for missing/incomplete final-results artifacts.
## Missing layers are grouped into one job per dataset/model.
significance-missing:
	@set -eu
	mkdir -p logs
	missing="$$($(SIGNIFICANCE_CHECK) --results-root results --models $(SIGNIFICANCE_MODELS) --missing)"
	if [ -z "$$missing" ]; then
		echo "All required significance Artifacts are complete."
		exit 0
	fi
	grouped="$$(printf '%s\n' "$$missing" | awk -F '\t' '
		function emit() {
			if (dataset == "") return
			if (layers == "none") print dataset "\t" model "\t" layers
			else print dataset "\t" model "\t[" layers "]"
		}
		$$1 != dataset || $$2 != model {
			emit()
			dataset = $$1
			model = $$2
			layers = $$3
			next
		}
		{ layers = layers "," $$3 }
		END { emit() }
	')"
	printf '%s\n' "$$grouped" | while IFS="$$(printf '\t')" read -r dataset model layers; do
		case "$$dataset" in
			podcast) config="configs/significance/$$model/significance.yml"; override="--ieeg.encoding.per_model_config.layer=$$layers"; label="significance-podcast" ;;
			brain-treebank-cars-2) config="configs/brain_treebank/cars_2/significance/$$model/significance.yml"; override="--ieeg.encoding.per_model_config.layer=$$layers"; label="significance-cars-2" ;;
			deep-fmri) config="configs/cross_dataset/fmri/significance/$$model.yml"; override="--fmri.layer=$$layers"; label="significance-fmri" ;;
			*) echo "Unsupported missing-significance dataset '$$dataset'" >&2; exit 2 ;;
		esac
		[ "$$layers" != "none" ] || override=""
		[ -f "$$config" ] || { echo "Missing significance config: $$config" >&2; exit 2; }
		echo "Submitting missing significance for $$dataset / $$model / layers $$layers"
		if [ "$(CMD)" = "sbatch" ]; then
			$(SBATCH) --job-name="$$label-$$model-$(USR)-$(DT)" submit.sh scripts/run_significance.py --config "$$config" $$override $(OVERRIDES)
		else
			$(CMD) scripts/run_significance.py --config "$$config" $$override $(OVERRIDES)
		fi
	done

## Run one RRR mode once per selected model. Leave-one-subject-out runs once per
## selected model layer.
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
	[ -n "$$models" ] || { echo "MODELS contains no models supported by $(DATASET) RRR" >&2; exit 2; }
	for model in $$models; do
		if [ "$(DATASET)" = "fmri" ] || [ "$$mode" = "cross_dataset" ]; then
			config="$$config_root/$$model.yml"
		else
			config="$$config_root/$$model/$$mode.yml"
		fi
		[ -f "$$config" ] || { echo "Missing RRR config: $$config" >&2; exit 2; }
		if [ "$$mode" = "leave_one_subject_out" ]; then
			layers="$$(uv run python -c 'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1])) or {}; layer = payload.get("ieeg", {}).get("encoding", {}).get("per_model_config", {}).get("layer"); print("none" if layer is None else " ".join(map(str, layer if isinstance(layer, list) else [layer])))' "$$config")"
			[ -n "$$layers" ] || { echo "Leave-one-subject-out config has no model layers: $$config" >&2; exit 2; }
		fi
	done
	for model in $$models; do
		if [ "$(DATASET)" = "fmri" ] || [ "$$mode" = "cross_dataset" ]; then
			config="$$config_root/$$model.yml"
		else
			config="$$config_root/$$model/$$mode.yml"
		fi
		if [ "$$mode" = "leave_one_subject_out" ]; then
			layers="$$(uv run python -c 'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1])) or {}; layer = payload.get("ieeg", {}).get("encoding", {}).get("per_model_config", {}).get("layer"); print("none" if layer is None else " ".join(map(str, layer if isinstance(layer, list) else [layer])))' "$$config")"
			for layer in $$layers; do
				layer_override=""
				layer_label="none"
				if [ "$$layer" != "none" ]; then
					layer_override="--ieeg.encoding.per_model_config.layer=$$layer"
					layer_label="$$layer"
				fi
				echo "Running $$mode RRR for $(DATASET) / $$model / layer $$layer_label: $$config"
				$(call launch,rrr-$(DATASET)-$$mode-layer-$$layer_label,scripts/run_rrr.py,$$layer_override)
			done
		else
			echo "Running $$mode RRR for $(DATASET) / $$model: $$config"
			$(call launch,rrr-$(DATASET)-$$mode,scripts/run_rrr.py)
		fi
	done

## Run directed cross-condition comparisons once per selected model layer.
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
		layers="$$(uv run python -c 'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1])) or {}; layer = payload.get("per_model_config", {}).get("layer"); print("none" if layer is None else " ".join(map(str, layer if isinstance(layer, list) else [layer])))' "$$config")"
		[ -n "$$layers" ] || { echo "Cross-condition config has no model layers: $$config" >&2; exit 2; }
	done
	for model in $(MODELS); do
		config="$$config_root/$$model.yml"
		layers="$$(uv run python -c 'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1])) or {}; layer = payload.get("per_model_config", {}).get("layer"); print("none" if layer is None else " ".join(map(str, layer if isinstance(layer, list) else [layer])))' "$$config")"
		for layer in $$layers; do
			layer_override=""
			layer_label="none"
			if [ "$$layer" != "none" ]; then
				layer_override="--per_model_config.layer=$$layer"
				layer_label="$$layer"
			fi
			echo "Running $(CROSS_MODE) comparison for $(DATASET) / $$model / layer $$layer_label: $$config"
			if [ "$(CMD)" = "sbatch" ]; then
				$(SBATCH) --job-name="cross-condition-$(CROSS_MODE)-$$model-layer-$$layer_label-$(USR)-$(DT)" submit.sh \
					scripts/run_cross_condition_rrr.py --config "$$config" $(OVERRIDES) $$layer_override
			else
				$(CMD) scripts/run_cross_condition_rrr.py --config "$$config" $(OVERRIDES) $$layer_override
			fi
		done
	done

## Submit the complete final-results RRR matrix after checking significance.
## Cross-dataset jobs sourced from fMRI wait for every internal fMRI job.
## Use CMD="uv run python" to execute the same matrix sequentially and locally.
final-results:
	@set -eu
	mkdir -p logs
	models="$(FINAL_MODELS)"
	[ -n "$$models" ] || { echo "FINAL_MODELS must contain at least one model" >&2; exit 2; }
	for model in $$models; do
		for config in \
			"configs/rrr/$$model/cross_model_layer.yml" \
			"configs/brain_treebank/cars_2/rrr/$$model/cross_model_layer.yml" \
			"configs/fmri/within_fmri_cross_model_layer/$$model.yml" \
			"configs/rrr/$$model/leave_one_subject_out.yml" \
			"configs/brain_treebank/cars_2/rrr/$$model/leave_one_subject_out.yml" \
			"configs/cross_condition/region/$$model.yml" \
			"configs/cross_condition/stimulus/$$model.yml" \
			"configs/cross_dataset/podcast/from_fmri/$$model.yml" \
			"configs/cross_dataset/cars-2/from_fmri/$$model.yml"; do
			[ -f "$$config" ] || { echo "Missing final-results config: $$config" >&2; exit 2; }
		done
	done
	$(SIGNIFICANCE_CHECK) --results-root results --models $(SIGNIFICANCE_MODELS)
	if [ "$(CMD)" = "sbatch" ]; then
		submit_job() {
			label="$$1"; model="$$2"; script="$$3"; config="$$4"; dependency="$${5:-}"
			if [ -n "$$dependency" ]; then
				job_id="$$($(SBATCH) --parsable --dependency="afterok:$$dependency" \
					--job-name="$$label-$$model-$(USR)-$(DT)" submit.sh \
					"$$script" --config "$$config" $(OVERRIDES))"
			else
				job_id="$$($(SBATCH) --parsable \
					--job-name="$$label-$$model-$(USR)-$(DT)" submit.sh \
					"$$script" --config "$$config" $(OVERRIDES))"
			fi
			job_id="$${job_id%%;*}"
			[ -n "$$job_id" ] || { echo "No Slurm job ID returned for $$label / $$model" >&2; exit 2; }
			echo "Submitted $$label / $$model as $$job_id" >&2
			printf '%s' "$$job_id"
		}
		fmri_dependencies=""
		for model in $$models; do
			submit_job internal-podcast "$$model" scripts/run_rrr.py "configs/rrr/$$model/cross_model_layer.yml" >/dev/null
			submit_job internal-cars-2 "$$model" scripts/run_rrr.py "configs/brain_treebank/cars_2/rrr/$$model/cross_model_layer.yml" >/dev/null
			fmri_id="$$(submit_job internal-fmri "$$model" scripts/run_rrr.py "configs/fmri/within_fmri_cross_model_layer/$$model.yml")"
			fmri_dependencies="$${fmri_dependencies:+$$fmri_dependencies:}$$fmri_id"
			submit_job loso-podcast "$$model" scripts/run_rrr.py "configs/rrr/$$model/leave_one_subject_out.yml" >/dev/null
			submit_job loso-cars-2 "$$model" scripts/run_rrr.py "configs/brain_treebank/cars_2/rrr/$$model/leave_one_subject_out.yml" >/dev/null
			submit_job cross-condition-region "$$model" scripts/run_cross_condition_rrr.py "configs/cross_condition/region/$$model.yml" >/dev/null
			submit_job cross-condition-stimulus "$$model" scripts/run_cross_condition_rrr.py "configs/cross_condition/stimulus/$$model.yml" >/dev/null
		done
		for model in $$models; do
			submit_job from-fmri-to-podcast "$$model" scripts/run_rrr.py "configs/cross_dataset/podcast/from_fmri/$$model.yml" "$$fmri_dependencies" >/dev/null
			submit_job from-fmri-to-cars-2 "$$model" scripts/run_rrr.py "configs/cross_dataset/cars-2/from_fmri/$$model.yml" "$$fmri_dependencies" >/dev/null
		done
	else
		for model in $$models; do
			$(CMD) scripts/run_rrr.py --config "configs/rrr/$$model/cross_model_layer.yml" $(OVERRIDES)
			$(CMD) scripts/run_rrr.py --config "configs/brain_treebank/cars_2/rrr/$$model/cross_model_layer.yml" $(OVERRIDES)
			$(CMD) scripts/run_rrr.py --config "configs/fmri/within_fmri_cross_model_layer/$$model.yml" $(OVERRIDES)
			$(CMD) scripts/run_rrr.py --config "configs/rrr/$$model/leave_one_subject_out.yml" $(OVERRIDES)
			$(CMD) scripts/run_rrr.py --config "configs/brain_treebank/cars_2/rrr/$$model/leave_one_subject_out.yml" $(OVERRIDES)
			$(CMD) scripts/run_cross_condition_rrr.py --config "configs/cross_condition/region/$$model.yml" $(OVERRIDES)
			$(CMD) scripts/run_cross_condition_rrr.py --config "configs/cross_condition/stimulus/$$model.yml" $(OVERRIDES)
		done
		for model in $$models; do
			$(CMD) scripts/run_rrr.py --config "configs/cross_dataset/podcast/from_fmri/$$model.yml" $(OVERRIDES)
			$(CMD) scripts/run_rrr.py --config "configs/cross_dataset/cars-2/from_fmri/$$model.yml" $(OVERRIDES)
		done
	fi

.PHONY: significance significance-missing missing-significance rrr cross-condition final-results

missing-significance: significance-missing
