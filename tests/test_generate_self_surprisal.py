import numpy as np

from scripts.utilities.generate_self_surprisal import _build_clean_context_batch


def test_clean_context_batch_masks_padding_and_caps_real_history():
    token_ids = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    batch, mask, positions = _build_clean_context_batch(
        token_ids, np.array([1, 3, 4]), context_len=3, pad_id=0
    )

    assert batch.tolist() == [[0, 0, 10], [10, 11, 12], [11, 12, 13]]
    assert mask.tolist() == [[0, 0, 1], [1, 1, 1], [1, 1, 1]]
    assert positions.tolist() == [[0, 0, 0], [0, 1, 2], [0, 1, 2]]
