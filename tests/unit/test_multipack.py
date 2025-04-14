from instructlab.training.multipack_sampler import (
    MultipackDistributedBatchSampler,
    PaddingDistributedBatchSampler,
)
import numpy as np
import pytest


MAX_LENGTH = 32768
NUM_REPLICAS = 8


@pytest.fixture
def lengths_data():
    # Attempted to approximate the lengths of the alpaca dataset using an exponential distribution
    # divide lengths by 4 because each token is approx. 4 chars on average
    return np.array(np.random.exponential(scale=500, size=50000) // 4, dtype=np.int32)


def verify_multipack_batches(
    batches_by_rank: list[
        list[np.array]
    ],  # [[np.array(batch) for batch in batches] for batches in rank]
    lengths: list[int],
    max_tokens: int,
    num_replicas: int,
    padding: bool = False,
):
    assert len(batches_by_rank) == num_replicas, (
        f"Expected {num_replicas} sets of batches, received {len(batches_by_rank)}"
    )
    # Cat batches from different ranks for easy batch checking
    all_batches = sum(batches_by_rank, start=[])

    # check each batch is within `max_tokens` size limit
    for b in all_batches:
        if padding:
            # All entries in batch will be padded to match longest entry in batch
            assert max(lengths[idx] for idx in b) * len(b) <= max_tokens, (
                "Padded batch * sequence exceeds max_tokens"
            )
        else:
            assert sum(lengths[idx] for idx in b) <= max_tokens, (
                "batch size exceeds max_tokens"
            )

    # check all ranks have same number of batches
    for batches in batches_by_rank[1:]:
        assert len(batches) == len(batches_by_rank[0]), (
            "Unequal number of batches on different ranks"
        )

    # check samples longer than max_tokens are excluded from batches
    all_chosen_idx = np.concatenate(all_batches)
    n_samples = len(lengths)
    not_selected = set(range(n_samples)) - set(all_chosen_idx)
    over_limit_indices = np.arange(n_samples)[lengths > max_tokens]
    assert set(over_limit_indices).issubset(not_selected), (
        "samples with length > max_tokens should be excluded from batches"
    )

    # check all tokens are allocated to batches except for at most (num_replicas -1) * max_tokens.
    # This is because the last set of batches will be dropped if there aren't num_replicas batches
    not_selected_not_over = set(not_selected) - set(over_limit_indices)
    total_remaining_tokens = sum(lengths[idx] for idx in not_selected_not_over)
    assert total_remaining_tokens <= (num_replicas - 1) * max_tokens, (
        "Tokens remaining that should have been allocated to machines"
    )


def test_multipack_distributed_batch_sampler(lengths_data):
    batches_by_rank = [
        MultipackDistributedBatchSampler(
            MAX_LENGTH, lengths_data, num_replicas=NUM_REPLICAS, rank=rank
        ).generate_batches()
        for rank in range(NUM_REPLICAS)
    ]

    verify_multipack_batches(
        batches_by_rank, lengths_data, MAX_LENGTH, NUM_REPLICAS, padding=False
    )


def test_padding_distributed_batch_sampler(lengths_data):
    batches_by_rank = [
        PaddingDistributedBatchSampler(
            MAX_LENGTH, lengths_data, num_replicas=NUM_REPLICAS, rank=rank
        ).generate_batches()
        for rank in range(NUM_REPLICAS)
    ]

    verify_multipack_batches(
        batches_by_rank, lengths_data, MAX_LENGTH, NUM_REPLICAS, padding=True
    )
