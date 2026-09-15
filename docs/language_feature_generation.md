# Language Feature Generation

`scripts/utilities/generate_language_features.py` creates reusable annotations before
language decoding. It supports Podcast, every Brain Treebank movie, and story-batched
deep-fMRI TextGrid transcripts. The
Podcast decoder registry reads useful generated columns directly from `tokens.tsv`, so
decoding does not rerun parsers or lexicon joins.

## Architecture and artifacts

Each dataset loader returns a `StimulusDocument` with contiguous canonical word IDs,
surface text, timing, sentence membership, optional speakers, source IDs, and media
references. Providers consume this contract and may have dataset-specific overrides.
Deep-fMRI generation uses one document and artifact directory per story. Run
`scripts/utilities/generate_fmri_language_boundaries.py` first: TED2020-corrupted English
Segment Any Text supplies sentence boundaries for the unpunctuated word tier, while PSST
supplies audio-derived IU boundaries. Both are snapped back to the exact TextGrid words.
The boundary and feature sweeps resume at the story-artifact level.
`scripts/utilities/generate_surprisal.py` updates only the shared
GPT-2 surprisal and entropy columns in existing artifacts, and creates canonical minimal artifacts
when a story has no completed full-provider output. It is resumable by model checksum
and avoids rerunning unrelated audio and document-level providers.
`scripts/utilities/generate_self_surprisal.py` creates
separate model/dataset/stimulus artifacts under
`processed_data/language_features/self_surprisal/`. Each target receives all available
real preceding tokens up to the configured context limit. Left padding is excluded by
the attention mask, real-context positions restart at zero, and the first target is
missing because it has no linguistic context. Subtoken surprisal is summed within each
word in bits, while word entropy is the first subtoken's predictive entropy in nats.
Causal models use their configured tokenizer, final LM head, and initialization. Target
windows are batched without changing their contexts, and the batch size halves
automatically on CUDA OOM. The random GPT-2 XL uses its configured seed. RoBERTa uses
the same procedure with GPT-2 small and records that fallback in `manifest.json`.
Generate these model-specific values by invoking
`scripts/utilities/generate_self_surprisal.py` with the desired config.
`scripts/utilities/generate_fmri_iu_features.py` similarly
populates IU position, onset, end, and length from the already aligned PSST boundaries
without rerunning audio inference or unrelated providers.
`scripts/utilities/generate_fmri_interpretable_features.py` adds the shared decoding subset from native timing
and sentence structure, Stanza UD syntax, and lexical norms without invoking the heavy
diarization or document-level coreference providers.

Negative or missing source sentence IDs are normalized by assigning each contiguous run
of affected words its own synthetic sentence ID after the source's nonnegative ID range.
For the centered `[-1, 1]` NRC-VAD release, emotional intensity is absolute valence.

Providers are resource bundles rather than one function per target. The built-ins are:

- `native`: timing, structural controls, and source annotations under `native_*` names;
- `gpt2_surprisal`: standardized GPT-2-small word surprisal in bits and first-subtoken
  predictive entropy in nats, resetting at each stimulus and using up to 1,024
  preceding BPE tokens; the first word is missing because it has no preceding
  linguistic context;
- `ud`: common Stanza EWT lemmas, UD POS/morphology/dependencies, deterministic tree
  measures, and a parser-token-to-canonical-word crosswalk;
- `lexical_norms`: configuration-driven norm tables, longest non-overlapping phrase
  matching, explicit availability flags, Lancaster composites, and VAD intensity;
- `swow`: association response degree and entropy;
- `wordnet`: sense count and explicitly named most-frequent-sense (`*_mfs`) outputs.
- `disfluency`: conservative filled-pause, repetition, discourse-marker, and lexical
  recurrence labels derived deterministically from canonical words;
- `speaker_diarization`: local pyannote Community-1 exclusive diarization, anonymous
  speaker labels, and greatest-overlap alignment to canonical words;
- `entity_coreference`: Stanza OntoNotes NER and GUM no-speaker document-level
  coreference, with token labels plus entity/mention spans and antecedent relations.

One stimulus directory contains `tokens.tsv`, `spans.tsv`, `relations.tsv`,
`provider_tokens.tsv`, `segments.tsv`, and `manifest.json`. `segments.tsv` retains
provider-owned timed media annotations such as diarized speaker turns. The manifest
records the configuration, input fingerprint, resource/model checksums, provider
versions, feature ownership, and coverage. Writes are atomic. Existing artifacts are
reused only when their fingerprint matches; changed inputs require `refresh: true`.

Brain Treebank native syntax/acoustics are retained separately from common annotations.
Stanza tokenizes reconstructed sentences normally and is aligned back by character span.
For punctuation or contractions that yield multiple parser words per canonical word, the
word-level label comes from the syntactic head component and the full crosswalk remains in
`provider_tokens.tsv`.

## Podcast decoding-task inventory

All artifact-backed targets below are complete. Native labels preserve the Podcast
source annotations, while standardized labels come from the shared generation pipeline
(notably Stanza UD). Raw Lancaster dimensions remain separate from the aggregate
sensorimotor strength and exclusivity composites. Categorical missing values are not a
class: they remain missing and are excluded by the decoder's finite-target mask.

## Task-to-subspace reconstruction

`scripts/language-interpret/run_language_reconstruction.py` uses any configured subset of the same
task registry as predictors of the RRR and PCA factors in one selected RRR Artifact.
Unlike language decoding, missing task values are retained: continuous values
receive fold-local median imputation plus missingness indicators, while categorical
missingness is an explicit level. Requested tasks with no values in the selected dataset
fail during preflight rather than silently changing the predictor set.

The executable resolves the RRR and significance inputs as Artifacts and writes
reconstruction fields directly to its output Artifact; reconstruction has no direct
result-directory loading or saving interface.

A fold-local multi-output ridge model jointly predicts every coordinate at each
configured rank. Both raw weighted factors and QR-orthonormalized spans are evaluated.
Because source-fold axes can rotate in cross-dataset mode, its weighted factor is derived
from the leading eigensystem of the mean source-fold `A A^T` operator. Predictor sets
can evaluate the registered language tasks, same-word GloVe vectors, or their combined
design. If a GloVe condition is requested, every condition is restricted to the same
GloVe-covered words and the saved source folds are remapped without changing fold
identity. Results include absolute metrics, RRR-minus-PCA comparisons, and paired
GloVe/task/combined contrasts. See `configs/rrr/language_reconstruction.yml` for the
complete interface.

| Task name | Kind | Measurement | Source | Status |
|---|---|---|---|---|
| `speaker_identity` | categorical | Anonymous within-episode speaker cluster | pyannote Community-1 | Complete |
| `native_part_of_speech` | categorical | Source POS tag | Podcast native | Complete |
| `native_surprisal` | continuous | Source LM surprisal | Podcast native | Complete |
| `native_entropy` | continuous | Source LM entropy | Podcast native | Complete |
| `part_of_speech` | categorical | Standardized universal POS | Stanza UD | Complete |
| `fine_part_of_speech` | categorical | Standardized language-specific POS | Stanza UD | Complete |
| `content_word` | categorical | UD content/function indicator | Stanza UD | Complete |
| `surprisal` | continuous | GPT-2-small surprisal in bits | GPT-2 provider | Complete |
| `standardized_entropy` | continuous | GPT-2-small first-subtoken predictive entropy in nats | GPT-2 provider | Complete |
| `self_surprisal` | continuous | Embedding model's own surprisal in bits; GPT-2 fallback for RoBERTa | Per-model artifact | Complete |
| `self_entropy` | continuous | Embedding model's first-subtoken predictive entropy in nats; GPT-2 fallback for RoBERTa | Per-model artifact | Complete |
| `entropy` | continuous | Legacy source entropy alias | Podcast native | Complete (legacy) |
| `surprisal_class` | categorical | Three-way surprisal bin | Legacy Podcast CSV | Complete (legacy) |
| `story_position` | continuous | Normalized word progress | Native provider | Complete |
| `story_time_position` | continuous | Normalized elapsed-time progress | Native provider | Complete |
| `sentence_position` | continuous | Normalized within-sentence position | Native provider | Complete |
| `sentence_onset` | categorical | First word of sentence | Native provider | Complete |
| `sentence_end` | categorical | Last word of sentence | Native provider | Complete |
| `sentence_length` | continuous | Sentence length in words | Native provider | Complete |
| `iu_position` | continuous | Normalized within-IU position | Native provider | Complete |
| `iu_onset` | categorical | First word of intonation unit | Native provider | Complete |
| `iu_end` | categorical | Last word of intonation unit | Native provider | Complete |
| `iu_length` | continuous | Intonation-unit length in words | Native provider | Complete |
| `pre_word_pause` | continuous | Silence before word | Native provider | Complete |
| `post_word_pause` | continuous | Silence after word | Native provider | Complete |
| `word_duration` | continuous | Word duration | Native provider | Complete |
| `word_length` | continuous | Orthographic length | Native provider | Complete |
| `word_zipf_frequency` | continuous | Zipf lexical frequency | wordfreq | Complete |
| `concreteness` | continuous | Concreteness norm | Lexical norms | Complete |
| `age_of_acquisition` | continuous | Acquisition-age norm | Lexical norms | Complete |
| `word_prevalence` | continuous | Population prevalence norm | Lexical norms | Complete |
| `semantic_diversity` | continuous | Contextual diversity norm | Lexical norms | Complete |
| `socialness` | continuous | Socialness norm | Lexical norms | Complete |
| `valence` | continuous | Valence rating | NRC-VAD | Complete |
| `arousal` | continuous | Arousal rating | NRC-VAD | Complete |
| `dominance` | continuous | Dominance rating | NRC-VAD | Complete |
| `emotional_intensity` | continuous | Absolute centered valence | NRC-VAD composite | Complete |
| `visual_strength` | continuous | Raw visual strength | Lancaster | Complete |
| `auditory_strength` | continuous | Raw auditory strength | Lancaster | Complete |
| `haptic_strength` | continuous | Raw haptic strength | Lancaster | Complete |
| `olfactory_strength` | continuous | Raw olfactory strength | Lancaster | Complete |
| `gustatory_strength` | continuous | Raw gustatory strength | Lancaster | Complete |
| `interoceptive_strength` | continuous | Raw interoceptive strength | Lancaster | Complete |
| `hand_arm_action_strength` | continuous | Raw hand/arm action strength | Lancaster | Complete |
| `mouth_throat_action_strength` | continuous | Raw mouth/throat action strength | Lancaster | Complete |
| `foot_leg_action_strength` | continuous | Raw foot/leg action strength | Lancaster | Complete |
| `head_action_strength` | continuous | Raw head action strength | Lancaster | Complete |
| `torso_action_strength` | continuous | Raw torso action strength | Lancaster | Complete |
| `sensorimotor_strength` | continuous | Maximum sensorimotor strength composite | Lancaster composite | Complete |
| `sensorimotor_exclusivity` | continuous | Sensorimotor exclusivity composite | Lancaster composite | Complete |
| `association_degree` | continuous | Number of association responses | SWOW | Complete |
| `association_entropy` | continuous | Association-response entropy | SWOW | Complete |
| `sense_count` | continuous | Number of WordNet senses | WordNet | Complete |
| `wordnet_lexname_mfs` | categorical | Most-frequent-sense lexname | WordNet MFS | Complete |
| `person_relatedness_mfs` | categorical | MFS person-related indicator | WordNet MFS | Complete |
| `animacy_mfs` | categorical | MFS animacy class | WordNet MFS | Complete |
| `motion_concept_mfs` | categorical | MFS motion-concept indicator | WordNet MFS | Complete |
| `cognition_concept_mfs` | categorical | MFS cognition-concept indicator | WordNet MFS | Complete |
| `communication_concept_mfs` | categorical | MFS communication-concept indicator | WordNet MFS | Complete |
| `perception_concept_mfs` | categorical | MFS perception-concept indicator | WordNet MFS | Complete |
| `number_of_hypernyms_mfs` | continuous | MFS hypernym count | WordNet MFS | Complete |
| `ontology_depth_mfs` | continuous | MFS ontology depth | WordNet MFS | Complete |
| `dependency_relation` | categorical | UD dependency relation | Stanza UD | Complete |
| `tense` | categorical | UD tense | Stanza UD | Complete |
| `mood` | categorical | UD mood | Stanza UD | Complete |
| `voice` | categorical | UD voice | Stanza UD | Complete |
| `grammatical_person` | categorical | UD grammatical person | Stanza UD | Complete |
| `number` | categorical | UD grammatical number | Stanza UD | Complete |
| `definiteness` | categorical | UD definiteness | Stanza UD | Complete |
| `dependency_depth` | continuous | Token depth in dependency tree | UD-derived | Complete |
| `dependency_length` | continuous | Token-to-head distance | UD-derived | Complete |
| `head_direction` | categorical | Root/left/right head direction | UD-derived | Complete |
| `number_of_dependents` | continuous | Direct dependent count | UD-derived | Complete |
| `subtree_size` | continuous | Dependency subtree size | UD-derived | Complete |
| `left_dependents` | continuous | Left dependent count | UD-derived | Complete |
| `right_dependents` | continuous | Right dependent count | UD-derived | Complete |
| `open_dependencies` | continuous | Open dependency arcs | UD-derived | Complete |
| `dependency_locality_cost` | continuous | Dependency-locality integration cost | UD-derived | Complete |
| `filled_pause` | categorical | Conservative filled-pause inventory | Disfluency | Complete |
| `immediate_word_repetition` | categorical | Same normalized word as previous token | Disfluency | Complete |
| `repeated_phrase` | categorical | Second realization of adjacent repeated 2–5-word phrase | Disfluency | Complete |
| `discourse_marker` | categorical | Conservative contextual discourse marker | Disfluency | Complete |
| `discourse_marker_type` | categorical | Detected marker identity | Disfluency | Complete |
| `word_seen_before` | categorical | Normalized word occurred earlier in story | Recurrence | Complete |
| `prior_word_count` | continuous | Number of earlier occurrences | Recurrence | Complete |
| `distance_from_previous_occurrence` | continuous | Tokens since previous occurrence | Recurrence | Complete |
| `word_in_previous_5` | categorical | Word occurs among previous five tokens | Recurrence | Complete |
| `is_named_entity` | categorical | Word belongs to a named entity | Stanza NER | Complete |
| `named_entity_type` | categorical | OntoNotes entity type, entity words only | Stanza NER | Complete |
| `is_coreference_mention` | categorical | Word belongs to a predicted mention | Stanza coreference | Complete |
| `coreference_newness` | categorical | First/subsequent mention in chain | Stanza coreference | Complete |
| `coreference_mention_form` | categorical | Proper/pronominal/nominal mention form | Stanza coreference + UD | Complete |
| `coreference_mention_order` | continuous | One-based order in chain | Stanza coreference | Complete |
| `coreference_chain_size` | continuous | Predicted mentions in chain | Stanza coreference | Complete |
| `coreference_antecedent_distance_words` | continuous | Distance from preceding chain mention | Stanza coreference | Complete |

Bookkeeping fields, availability flags, matching provenance, token/head IDs, `lemma`,
the combined morphology string, and `aspect` are intentionally not decoding targets.

## Remaining decoding tasks

- `aspect` is generated but unusable for Podcast decoding because its coverage is zero.
- Dominant modality, contextual WordNet sense and contextual verb class are not yet
  implemented.
- Semantic-role labeling, entity salience, broader discourse, and sentence/IU or
  broader coherence targets are not yet implemented. Named entities and coreference
  now cover the initial type, mention-form, newness, order, distance, and chain-size tasks.
- Contextual prediction targets (contextual LM distributions, entropy change, semantic
  prediction error, and structured surprisal) are not yet implemented.
- Event-boundary and mental-state/perspective targets are not yet implemented.
- Expanded prosody beyond word timing and pauses (F0/intensity, rate, lengthening,
  pitch reset, prominence, ToBI, and repair/false-start targets) is not yet implemented.
- Brain Treebank artifacts can be loaded by the language decoder with dataset-specific
  task manifests. The retained fMRI decoder loads factors from a normal internal fMRI
  RRR Artifact, trains on training-story words, tunes with leave-one-session-out inner
  folds, and evaluates only on `wheretheressmoke`.

## Resource staging

Run:

```bash
uv run python scripts/utilities/generate_language_features.py \
  --config configs/language_features/podcast.yml --stage-resources
```

The Podcast speaker model is gated. Before staging, accept the conditions for
`pyannote/speaker-diarization-community-1` on Hugging Face and export an `HF_TOKEN`
with read access. The staged snapshot is then loaded locally; generation does not send
the Podcast audio to a hosted diarization service. By default the provider infers the
speaker count and selects CUDA when available. Its config supports `device`, an exact
`num_speakers`, or compatible `min_speakers`/`max_speakers` bounds. Exact and bounded
speaker-count settings cannot be combined.

Speaker clusters are named deterministically by first appearance (`SPEAKER_00`,
`SPEAKER_01`, ...). Each timed word receives the exclusive turn with the largest
positive overlap; exact ties go to the earlier turn and words without overlap remain
missing. Raw turns are saved in `segments.tsv`, while the manifest reports speaker
durations, word counts, labeled-word coverage, and unmatched words for audit. These are
anonymous identities within this episode, not real-person or cross-episode identities.

Generation never downloads missing resources. Staging can download configured Stanza or
NLTK model assets and URL-based resources, and validates every configured checksum.
Human norm files marked `manual: true` must be placed at their configured paths. This
keeps license/click-through requirements explicit. The example mappings should be checked
against the exact release columns and populated with release checksums before analysis.

Lexical matching is POS-aware when a resource supplies a POS column. Otherwise it tries
exact surface, lowercase surface, then lowercase lemma. Multiword entries use the longest
non-overlapping match within a sentence, are retained as spans, and propagate values to
their member words. Human ratings remain missing when unmatched and always have a paired
`<feature>_available` column; the generator performs no imputation.

## Feasibility catalog

Scores are end-to-end implementation feasibility: **5** deterministic lookup/derivation,
**4** established model with manageable alignment, **3** substantial model/alignment
work, **2** research-grade or fragile tooling, and **1** mostly manual/underspecified.
Readiness is **R** already in the repository, **G** generatable from present inputs after
staging, and **M** missing media/manual annotation.

| Family | Included tasks | Score | Podcast | Brain Treebank |
|---|---|---:|:---:|:---:|
| Existing controls | POS, content word, frequency, length/duration, pauses, sentence/story position | 5 | R | R |
| Existing LM controls | surprisal, entropy, fine POS | 4–5 | R | R/G |
| IU structure | IU onset/end/length/position | 5 | R | M |
| Core lexical norms | concreteness, AoA, prevalence, VAD, emotional intensity, socialness | 5 | G | G |
| Semantic diversity | semantic diversity | 4 | G | G |
| Lancaster | all 11 dimensions, strength, exclusivity, dominant modality | 5 | G | G |
| Human associations | association degree and entropy | 5 | G | G |
| WordNet ambiguity | sense count | 5 | G | G |
| WordNet MFS classes | lexname, person, motion, cognition, communication, perception | 4 | G | G |
| WordNet hierarchy | animacy, hypernym count, ontology depth | 3–4 | G | G |
| Verb class / contextual sense | stative-eventive; context-selected WordNet variants | 2 | G | G |
| Direct UD | relations, POS, morphology, tense/aspect/mood/voice/person/number/definiteness | 4 | G | R/G |
| Derived dependency | depth/length/direction, dependents, subtree, open arcs, locality | 5 | G | R/G |
| Semantic roles | predicates, PropBank roles, role flags/counts/distances | 3 | G | G |
| Coreference | newness, mention order, antecedent distances, chain/recency/persistence | 2 | G | G |
| Mention form | mention form and pronoun/full-NP | 3 | G | G |
| Entity salience | composite salience | 2 | G | G |
| Named entities | entity type and person reference | 4 | G | R/G |
| Speaker reference | reference to current speaker/narrator | 1 | M | G |
| Discourse connective | connective detection | 4 | G | G |
| PDTB discourse | senses, arguments, explicitness, attribution, span variants | 2 | G | G |
| Sentence/IU coherence | preceding sentence or IU coherence | 4 | R/G | G/M |
| Broader coherence | word/local/global similarity and topic shift | 3 | G | G |
| Token semantic update | semantic update magnitude | 2 | G | G |
| Event boundaries | boundary strength and six situation-change components | 1 | M | M |
| Mental-state cues | lexical mental-state predicate | 4 | G | G |
| Contextual mental state | category, indirect speech, narration/dialogue | 3 | G | R/G |
| Direct speech | quoted speech | 4 | G | R/G |
| Perspective | holder and shift | 1 | M | M |
| Epistemic semantics | factivity and epistemic modality | 2 | G/M | G/M |
| Unigram LM | unigram surprisal | 5 | R | R |
| Causal LM distributions | contextual surprisal, entropy, rank, top-k mass | 4 | R/G | R/G |
| Entropy change | entropy reduction | 3 | G | G |
| Semantic prediction | semantic prediction error | 2 | G | G |
| Structured surprisal | syntactic and POS surprisal | 2 | G | G |
| Timing acoustics | pre/post-word pause | 5 | R | R |
| Basic F0/intensity | means, extrema, ranges, slopes | 4 | R/G | R native; M common |
| Rate | speaking rate | 4 | G | R/G |
| Articulation rate | pause-excluding rate | 3 | G | M |
| Residual lengthening | phrase-final lengthening | 2 | G | M |
| Pitch reset | IU/phrase pitch reset | 3 | G | M |
| Prominence | continuous prominence | 2 | G | M |
| ToBI | pitch accent and boundary tone | 1 | M | M |
| Disfluency | fluent/pause/repetition/repair/false-start label | 3 | G | G/M |
| Repair onset | self-correction onset | 2 | G/M | G/M |

The first milestone implements the framework and high-feasibility native, lexical,
WordNet, and UD families. SRL, coreference, discourse, narrative, contextual LM, and
expanded prosody remain later providers because they require further model and labeling
decisions.
