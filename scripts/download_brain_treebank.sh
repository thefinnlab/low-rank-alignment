#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/brain-treebank}"
MIN_MOVIES="${MIN_MOVIES:-2}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/download_brain_treebank.sh [--data-root PATH] [--min-movies N] [--dry-run]
EOF
}

while (($#)); do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --min-movies) MIN_MOVIES="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

BASE_URL="${BTB_BASE_URL:-https://braintreebank.dev/data}"
DOWNLOAD_DIR="$DATA_ROOT/.downloads"
mkdir -p "$DATA_ROOT" "$DOWNLOAD_DIR"

download_file() {
  local url="$1" destination="$2"
  if [[ -s "$destination" ]]; then echo "Using existing $destination"; return; fi
  echo "Downloading $url"
  curl --fail --location --retry 4 --continue-at - --output "$destination" "$url"
}

extracted_archive_is_complete() {
  local name="$1"
  case "$name" in
    subject_metadata.zip) [[ -n "$(find "$DATA_ROOT/subject_metadata" -type f -name 'sub_*_trial*_metadata.json' -print -quit 2>/dev/null)" ]] ;;
    subject_timings.zip) [[ -n "$(find "$DATA_ROOT/subject_timings" -type f -print -quit 2>/dev/null)" ]] ;;
    electrode_labels.zip) [[ -n "$(find "$DATA_ROOT/electrode_labels" -type f -print -quit 2>/dev/null)" ]] ;;
    localization.zip) [[ -n "$(find "$DATA_ROOT/localization" -type f -print -quit 2>/dev/null)" ]] ;;
    speaker_annotations.zip) [[ -n "$(find "$DATA_ROOT/speaker_annotations" -type f -print -quit 2>/dev/null)" ]] ;;
    scene_annotations.zip) [[ -s "$DATA_ROOT/scene_annotations.json" ]] ;;
    transcripts.zip) [[ -n "$(find "$DATA_ROOT/transcripts" -type f -print -quit 2>/dev/null)" ]] ;;
    trees.zip) [[ -n "$(find "$DATA_ROOT/trees" -type f -print -quit 2>/dev/null)" ]] ;;
    *) return 1 ;;
  esac
}

download_zip() {
  local name="$1" archive="$DOWNLOAD_DIR/$1"
  if extracted_archive_is_complete "$name"; then
    echo "Using existing extracted data for $name"
    return
  fi
  download_file "$BASE_URL/$name" "$archive"
  if ! unzip -tq "$archive" >/dev/null; then
    rm -f "$archive"
    download_file "$BASE_URL/$name" "$archive"
    unzip -tq "$archive" >/dev/null
  fi
  # Metadata must be extracted even during a dry run so eligibility can be
  # computed before any trial recordings are selected.
  if ((DRY_RUN == 0)) || [[ "$name" == "subject_metadata.zip" ]]; then
    unzip -oq "$archive" -d "$DATA_ROOT"
  fi
}

for archive in subject_metadata.zip subject_timings.zip electrode_labels.zip localization.zip speaker_annotations.zip scene_annotations.zip transcripts.zip trees.zip; do
  download_zip "$archive"
done
download_file "$BASE_URL/corrupted_elec.json" "$DATA_ROOT/corrupted_elec.json"

if [[ ! -d "$DATA_ROOT/subject_metadata" ]]; then
  echo "subject_metadata was not extracted to $DATA_ROOT/subject_metadata" >&2
  exit 1
fi

declare -A subject_movies=()
declare -A subject_trials=()
while IFS= read -r metadata_path; do
  filename="${metadata_path##*/}"
  subject="${filename#sub_}"; subject="${subject%%_trial*}"
  trial="${filename#*_trial}"; trial="${trial%%_metadata.json}"
  movie="$(jq -r '.filename // empty' "$metadata_path")"
  [[ -n "$movie" ]] || continue
  subject_movies["$subject"]+="${movie}"$'\n'
  subject_trials["$subject"]+="${trial}"$'\n'
done < <(find "$DATA_ROOT/subject_metadata" -type f -name 'sub_*_trial*_metadata.json' | sort)

eligible_subjects=()
for subject in "${!subject_movies[@]}"; do
  movie_count="$(printf '%s' "${subject_movies[$subject]}" | sed '/^$/d' | sort -u | wc -l)"
  if ((movie_count >= MIN_MOVIES)); then eligible_subjects+=("$subject"); fi
done
if ((${#eligible_subjects[@]})); then
  mapfile -t eligible_subjects < <(printf '%s\n' "${eligible_subjects[@]}" | sort -n)
fi

echo "Eligible subjects (at least $MIN_MOVIES movies): ${eligible_subjects[*]:-none}"
for subject in "${eligible_subjects[@]}"; do
  movies="$(printf '%s' "${subject_movies[$subject]}" | sed '/^$/d' | sort -u | paste -sd, -)"
  echo "  sub-$subject: $movies"
done

if ((DRY_RUN)); then echo "Dry run: no trial recordings extracted."; exit 0; fi

mkdir -p "$DATA_ROOT/all_subject_data"
for subject in "${eligible_subjects[@]}"; do
  while IFS= read -r trial; do
    [[ -n "$trial" ]] || continue
    basename="sub_${subject}_trial${trial}.h5"
    target="$DATA_ROOT/all_subject_data/$basename"
    if [[ -s "$target" ]]; then echo "Using existing $target"; continue; fi
    archive_name="$basename.zip"; archive="$DOWNLOAD_DIR/$archive_name"
    url="$BASE_URL/subject_data/sub_${subject}/trial${trial}/$archive_name"
    download_file "$url" "$archive"
    if ! unzip -tq "$archive" >/dev/null; then
      rm -f "$archive"
      download_file "$url" "$archive"
      unzip -tq "$archive" >/dev/null
    fi
    staging="$DOWNLOAD_DIR/extract-${subject}-${trial}"
    rm -rf "$staging"; mkdir -p "$staging"
    unzip -oq "$archive" -d "$staging"
    source="$(find "$staging" -type f -name "$basename" -print -quit)"
    if [[ -z "$source" ]]; then echo "Archive $archive did not contain $basename" >&2; exit 1; fi
    mv "$source" "$target"; rm -rf "$staging"
  done <<<"${subject_trials[$subject]}"
done

echo "Brain Treebank download complete: $DATA_ROOT"
