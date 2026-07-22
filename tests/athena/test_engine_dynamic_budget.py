from unittest.mock import patch

from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.utils import prepare_micro_batches


def test_prepare_micro_batches_uses_per_batch_token_budget():
    data = TensorDict({}, batch_size=(0,))
    tu.assign_non_tensor(data, use_dynamic_bsz=True, sp_size=2, max_token_len_per_gpu=3072)

    with patch("verl.workers.engine.utils.rearrange_micro_batches", return_value=([], [])) as rearrange:
        prepare_micro_batches(data)

    assert rearrange.call_args.kwargs["max_token_len"] == 6144
