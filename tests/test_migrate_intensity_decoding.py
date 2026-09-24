from scripts.utilities.migrate_intensity_decoding import (
    _is_matching_standard_variant, _variant_without_intensity,
)


def test_intensity_variant_component_is_removed_without_changing_other_components():
    assert _variant_without_intensity(
        "source/internal/podcast/offset-0/gpt2/layer-1/audio-intensity-v1/"
        "input-scaling-v1/label-offset-0"
    ) == (
        "source/internal/podcast/offset-0/gpt2/layer-1/input-scaling-v1/"
        "label-offset-0"
    )


def test_rank_only_variant_component_is_removed():
    assert _variant_without_intensity("audio-intensity-ranks-v1") == "default"


def test_standard_variant_can_include_speaker_task_component():
    intensity_variant = (
        "source/internal/podcast/offset-0/gpt2/layer-1/"
        "audio-intensity-ranks-v1/input-scaling-v1/aggregation-extrinsic/label-offset-0"
    )
    standard_variant = (
        "source/internal/podcast/offset-0/gpt2/layer-1/input-scaling-v1/"
        "speaker-labels-outer-fold-shared-other/aggregation-extrinsic/label-offset-0"
    )
    assert _is_matching_standard_variant(intensity_variant, standard_variant)
