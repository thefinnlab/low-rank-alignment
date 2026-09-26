.ONESHELL:

USR := $(shell whoami | head -c 2)
DT  := $(shell date +"%Y%m%d-%H:%M:%S")

# Shared controls. Override any of these on the make command line.
DATASET ?= podcast
MODELS ?= gpt2 gpt2-xl llama3-8b gpt-j-6b roberta-large random-gpt2 glove
FINAL_MODELS ?= gpt2 gpt2-xl gpt-j-6b roberta-large llama3-8b random-gpt2 glove
SIGNIFICANCE_MODELS := gpt2 gpt2-xl gpt-j-6b roberta-large llama3-8b random-gpt2 glove
RRR_MODE ?= cross_model_layer
SOURCE_DATASET ?=
CMD ?= sbatch
SBATCH ?= sbatch
OVERRIDES ?=
LAYER ?=
PER_LAYER ?= false
LANGUAGE_DECODING_CPUS ?= 1
FMRI_TR_PRE_FIR ?=
SIGNIFICANCE_CHECK ?= uv run python -m scripts.utilities.check_final_significance
LANGUAGE_DECODING_CHECK ?= uv run python -m scripts.utilities.check_language_decoding_results

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
## DATASET: podcast, cars-2, or fmri
## Example: make significance DATASET=podcast MODELS="gpt2 glove"
significance:
	@set -eu
	mkdir -p logs
	case "$(DATASET)" in
		podcast) config_root="configs/significance"; suffix="/significance.yml" ;;
		cars-2) config_root="configs/brain_treebank/cars_2/significance"; suffix="/significance.yml" ;;
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

## Run one RRR mode once per selected model. Leave-one-subject-out and fMRI
## cross_model_layer_per_layer run once per selected model layer.
## iEEG RRR_MODE: cross_model_layer, leave_one_subject_out, full_fit, or cross_dataset
## fMRI RRR_MODE: cross_model_layer,
##                cross_model_layer_per_layer, leave_one_subject_out, or cross_dataset
## cross_dataset also requires SOURCE_DATASET.
## Example: make rrr DATASET=fmri RRR_MODE=cross_model_layer_per_layer MODELS="gpt2 gpt2-xl"
rrr:
	@set -eu
	mkdir -p logs
	mode="$(RRR_MODE)"
	per_layer=false
	[ "$$mode" != "internal" ] || mode="cross_model_layer"
	case "$$mode" in
		cross_model_layer|cross_model_layer_per_layer|leave_one_subject_out|full_fit|cross_dataset) ;;
		*) echo "Unsupported RRR_MODE '$$mode'" >&2; exit 2 ;;
	esac
	case "$(DATASET)" in
		podcast) config_root="configs/rrr" ;;
		cars-2) config_root="configs/brain_treebank/cars_2/rrr" ;;
		fmri)
			case "$$mode" in
				cross_model_layer) config_root="configs/fmri/within_fmri_cross_model_layer" ;;
				cross_model_layer_per_layer) config_root="configs/fmri/within_fmri_cross_model_layer"; per_layer=true ;;
				leave_one_subject_out) config_root="configs/fmri/leave_one_subject_out"; per_layer=true ;;
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
		if [ "$$mode" = "leave_one_subject_out" ] && [ "$(DATASET)" != "fmri" ]; then
			layers="$$(uv run python -c 'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1])) or {}; layer = payload.get("ieeg", {}).get("encoding", {}).get("per_model_config", {}).get("layer"); print("none" if layer is None else " ".join(map(str, layer if isinstance(layer, list) else [layer])))' "$$config")"
			[ -n "$$layers" ] || { echo "Leave-one-subject-out config has no model layers: $$config" >&2; exit 2; }
		elif [ "$$per_layer" = true ]; then
			layers="$$(uv run python -c 'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1])) or {}; layer = payload.get("fmri", {}).get("layer"); print("none" if layer is None else " ".join(map(str, layer if isinstance(layer, list) else [layer])))' "$$config")"
			[ -n "$$layers" ] || { echo "fMRI per-layer config has no model layers: $$config" >&2; exit 2; }
		fi
	done
	for model in $$models; do
		if [ "$(DATASET)" = "fmri" ] || [ "$$mode" = "cross_dataset" ]; then
			config="$$config_root/$$model.yml"
		else
			config="$$config_root/$$model/$$mode.yml"
		fi
		if [ "$$mode" = "leave_one_subject_out" ] && [ "$(DATASET)" != "fmri" ]; then
			layers="$$(uv run python -c 'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1])) or {}; layer = payload.get("ieeg", {}).get("encoding", {}).get("per_model_config", {}).get("layer"); print("none" if layer is None else " ".join(map(str, layer if isinstance(layer, list) else [layer])))' "$$config")"
			for layer in $$layers; do
				layer_override=""
				layer_label="none"
				if [ "$$layer" != "none" ]; then
					layer_override="--ieeg.encoding.per_model_config.layer=[$$layer]"
					layer_label="$$layer"
				fi
				echo "Running $$mode RRR for $(DATASET) / $$model / layer $$layer_label: $$config"
				$(call launch,rrr-$(DATASET)-$$mode-layer-$$layer_label,scripts/run_rrr.py,$$layer_override)
			done
		elif [ "$$per_layer" = true ]; then
			layers="$$(uv run python -c 'import sys, yaml; payload = yaml.safe_load(open(sys.argv[1])) or {}; layer = payload.get("fmri", {}).get("layer"); print("none" if layer is None else " ".join(map(str, layer if isinstance(layer, list) else [layer])))' "$$config")"
			job_mode="$$mode"
			[ "$$mode" != "cross_model_layer_per_layer" ] || job_mode="cross_model_layer"
			for layer in $$layers; do
				layer_override=""
				layer_label="none"
				if [ "$$layer" != "none" ]; then
					layer_override="--fmri.layer=[$$layer]"
					layer_label="$$layer"
				fi
				echo "Running $$job_mode RRR for $(DATASET) / $$model / layer $$layer_label: $$config"
				$(call launch,rrr-$(DATASET)-$$job_mode-layer-$$layer_label,scripts/run_rrr.py,$$layer_override)
			done
		else
			echo "Running $$mode RRR for $(DATASET) / $$model: $$config"
			$(call launch,rrr-$(DATASET)-$$mode,scripts/run_rrr.py)
		fi
	done

## Decode target-dataset language labels from internal or directed source bases.
## Each selected model is one job by default; PER_LAYER=true submits one job per layer.
## Leave SOURCE_DATASET empty for internal decoding, or set it to another dataset.
language-decoding:
	@set -eu
	mkdir -p logs
	case "$(FMRI_TR_PRE_FIR)" in ""|true|false) ;; *) echo "FMRI_TR_PRE_FIR must be true or false" >&2; exit 2 ;; esac
	case "$(DATASET)" in podcast|cars-2|fmri) ;; *) echo "Unsupported language-decoding DATASET '$(DATASET)'" >&2; exit 2 ;; esac
	if [ -n "$(SOURCE_DATASET)" ]; then
		[ "$(SOURCE_DATASET)" != "$(DATASET)" ] || { echo "Leave SOURCE_DATASET empty for internal decoding" >&2; exit 2; }
		case "$(SOURCE_DATASET)" in podcast|cars-2|fmri) ;; *) echo "Unsupported language-decoding SOURCE_DATASET '$(SOURCE_DATASET)'" >&2; exit 2 ;; esac
		config="configs/language_decoding/cross_dataset/$(DATASET)/from_$(SOURCE_DATASET).yml"
		label="language-decoding-$(SOURCE_DATASET)-to-$(DATASET)"
	else
		config="configs/language_decoding/internal/$(DATASET).yml"
		label="language-decoding-$(DATASET)"
	fi
	[ -f "$$config" ] || { echo "Missing language-decoding config: $$config" >&2; exit 2; }
	fmri_tr_arg=""
	if [ "$(DATASET)" = "fmri" ] && [ -n "$(FMRI_TR_PRE_FIR)" ]; then fmri_tr_arg="--fmri_tr_pre_fir=$(FMRI_TR_PRE_FIR)"; fi
	[ -n "$(strip $(MODELS))" ] || { echo "MODELS must contain at least one model" >&2; exit 2; }
	case "$(PER_LAYER)" in true|false) ;; *) echo "PER_LAYER must be true or false" >&2; exit 2 ;; esac
	[ "$(PER_LAYER)" != "true" ] || [ -z "$(LAYER)" ] || { echo "PER_LAYER=true and LAYER cannot be used together" >&2; exit 2; }
	for model in $(MODELS); do
		layers="none"
		if [ "$(PER_LAYER)" = "true" ]; then
			layers="$$(uv run python -c 'import sys; from scripts.run_language_decoding_batch import build_configs; print(" ".join("none" if c.encoding.per_model_config.get("layer") is None else str(c.encoding.per_model_config.get("layer")) for c in build_configs(sys.argv[1], sys.argv[2])))' "$$config" "$$model")"
		elif [ -n "$(LAYER)" ]; then
			layers="$(LAYER)"
		fi
		for layer in $$layers; do
			layer_arg=""
			layer_label="none"
			if [ "$$layer" != "none" ]; then layer_arg="--layer $$layer"; layer_label="$$layer"; fi
			echo "Running $$label / $$model / layer $$layer_label: $$config"
			if [ "$(CMD)" = "sbatch" ]; then
				$(SBATCH) --cpus-per-task=$(LANGUAGE_DECODING_CPUS) --job-name="$$label-$$model-layer-$$layer_label-$(USR)-$(DT)" submit.sh scripts/run_language_decoding_batch.py --config "$$config" --model "$$model" $$layer_arg $$fmri_tr_arg $(OVERRIDES)
			else
				$(CMD) scripts/run_language_decoding_batch.py --config "$$config" --model "$$model" $$layer_arg $$fmri_tr_arg $(OVERRIDES)
			fi
		done
	done

## Submit one job per model/layer with any missing or incomplete internal or
## directed language-decoding Artifact. Every submitted run uses refresh=false.
language-decoding-missing:
	@set -eu
	mkdir -p logs
	case "$(FMRI_TR_PRE_FIR)" in ""|true|false) ;; *) echo "FMRI_TR_PRE_FIR must be true or false" >&2; exit 2 ;; esac
	case "$(OVERRIDES)" in *--fmri_tr_pre_fir=*) echo "Use FMRI_TR_PRE_FIR for language-decoding-missing" >&2; exit 2 ;; esac
	check_tr_arg=""
	if [ -n "$(FMRI_TR_PRE_FIR)" ]; then check_tr_arg="--fmri-tr-pre-fir=$(FMRI_TR_PRE_FIR)"; fi
	missing="$$($(LANGUAGE_DECODING_CHECK) --results-root results --models $(MODELS) $$check_tr_arg --missing)"
	if [ -z "$$missing" ]; then
		echo "All required language-decoding Artifacts are complete."
		exit 0
	fi
	printf '%s\n' "$$missing" | while IFS="$$(printf '\t')" read -r config source target model layer; do
		layer_arg=""
		layer_label="none"
		if [ "$$layer" != "none" ]; then layer_arg="--layer $$layer"; layer_label="$$layer"; fi
		if [ "$$source" = "$$target" ]; then
			label="language-decoding-$$target"
		else
			label="language-decoding-$$source-to-$$target"
		fi
		fmri_tr_arg=""
		if [ "$$target" = "fmri" ] && [ -n "$(FMRI_TR_PRE_FIR)" ]; then fmri_tr_arg="--fmri_tr_pre_fir=$(FMRI_TR_PRE_FIR)"; fi
		echo "Submitting missing $$label / $$model / layer $$layer_label: $$config"
		if [ "$(CMD)" = "sbatch" ]; then
			$(SBATCH) --cpus-per-task=$(LANGUAGE_DECODING_CPUS) --job-name="$$label-$$model-layer-$$layer_label-$(USR)-$(DT)" submit.sh scripts/run_language_decoding_batch.py --config "$$config" --model "$$model" $$layer_arg $$fmri_tr_arg $(OVERRIDES) --refresh=false
		else
			$(CMD) scripts/run_language_decoding_batch.py --config "$$config" --model "$$model" $$layer_arg $$fmri_tr_arg $(OVERRIDES) --refresh=false
		fi
	done

## Submit internal RRR, iEEG LOSO, and fMRI-to-iEEG transfer after checking significance.
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
		done
		for model in $$models; do
			$(CMD) scripts/run_rrr.py --config "configs/cross_dataset/podcast/from_fmri/$$model.yml" $(OVERRIDES)
			$(CMD) scripts/run_rrr.py --config "configs/cross_dataset/cars-2/from_fmri/$$model.yml" $(OVERRIDES)
		done
	fi

.PHONY: significance significance-missing missing-significance rrr final-results language-decoding language-decoding-missing missing-language-decoding

missing-significance: significance-missing
missing-language-decoding: language-decoding-missing

## Rebuild scaler-sensitive source artifacts, then all RRR/decoding consumers.
## Jobs are linked with Slurm afterok dependencies and this command waits for all.
regenerate-transfer-results:
	uv run python -m scripts.utilities.run_transfer_regeneration --models $(MODELS)

.PHONY: regenerate-transfer-results
