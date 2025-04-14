"""
MIT License

Copyright (c) 2023 One

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
taken from https://github.com/imoneoi/multipack_sampler
"""

# Standard
from typing import NamedTuple

# Third Party
from torch.utils.data import Sampler
import numpy as np
import torch
import math
from heapq import heapreplace, heappop, heappush
from numpy.typing import ArrayLike, NDArray
import torch.distributed as dist


def find_max_pack_len_with_padding(
    dataset,
    samples_per_minibatch,
    num_gpus,
    avg_sample_len,
    seed,
):
    """
    This function calculates the maximum batch length with padding for a given dataset. it uses a binary search to find the optimal addition to the average sample length that will result in the average batch size per minibatch being less than or equal to the number of samples per minibatch.

    Parameters:
    - dataset: The dataset for which the maximum batch length is to be calculated.
    - samples_per_minibatch: The number of samples per minibatch.
    - num_gpus: The number of GPUs available for computation.
    - avg_sample_len: The average length of a sample in the dataset.
    - seed: The seed for the random number generator.

    Returns:
    - The maximum batch length with padding for the given dataset.
    """

    def get_effective_samples_per_minibatch(num_tokens_per_gpu):
        """
        This nested function calculates the effective number of samples per minibatch for a given number of tokens per GPU.

        Parameters:
        - num_tokens_per_gpu: The number of tokens per GPU.

        Returns:
        - The effective number of samples per minibatch.

        The function creates a sampler using the MultipackDistributedBatchSampler class, generates batches using the sampler, and then returns the ratio of the dataset size to the number of batches.
        """
        sampler = PaddingDistributedBatchSampler(
            batch_max_length=num_tokens_per_gpu,
            lengths=dataset.get_lengths(),
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            seed=seed,
        )
        batches = sampler.generate_batches()
        return len(dataset) / len(batches)

    samples_per_gpu = samples_per_minibatch / num_gpus

    addition = int(avg_sample_len * 0.1 * samples_per_gpu)
    packing_max_batch_len = int(avg_sample_len * samples_per_gpu)

    avg_bs_per_minibatch = get_effective_samples_per_minibatch(
        packing_max_batch_len + addition
    )
    while avg_bs_per_minibatch <= samples_per_minibatch:
        addition *= 2
        avg_bs_per_minibatch = get_effective_samples_per_minibatch(
            packing_max_batch_len + addition
        )

    l = 0
    r = addition
    while r - l > 1:
        addition = (l + r) // 2
        avg_bs_per_minibatch = get_effective_samples_per_minibatch(
            packing_max_batch_len + addition
        )

        # check if simulation resulted in batch sizes close enough to goal and adjust if needed
        if abs(avg_bs_per_minibatch - samples_per_minibatch) <= max(
            10, round(avg_bs_per_minibatch * 0.02)
        ):
            break
        if avg_bs_per_minibatch > samples_per_minibatch:
            r = addition
        else:
            l = addition

    return packing_max_batch_len + addition


def find_packing_max_batch_len_and_grad_accum(
    num_gpus,
    avg_sample_len,
    effective_batch_size,
    max_batch_len_per_gpu,
    is_padding,
    dataset,
    seed,
):
    """
    Calculate the minimum gradient accumulation steps required and the corresponding maximum batch length.

    This function determines the minimum number of gradient accumulation steps needed to process the
    effective batch size within the constraints of the maximum batch length per GPU. It starts with
    the assumption of a single step (no accumulation) and increases the number of steps until the
    calculated batch length does not exceed the maximum allowed per GPU. The goal is to find the
    lowest gradient accumulation that allows fitting the batch within GPU limits, ensuring efficient
    utilization of computational resources.

    Parameters:
    - num_gpus (int): The number of GPUs over which the batch is distributed.
    - avg_sample_len (int): The average length of samples in the dataset, used to estimate batch length.
    - effective_batch_size (int): The total batch size intended to be processed across all GPUs and
      accumulation steps.
    - max_batch_len_per_gpu (int): The maximum permissible number of tokens on each GPU to avoid memory overflow.

    Returns:
    - Tuple[int, int]: A tuple where the first element is the maximum batch length that can be achieved
      without exceeding the per-GPU limit, and the second element is the minimum number of gradient
      accumulation steps required to maintain the effective batch size.
    """

    packing_max_batch_len = max_batch_len_per_gpu + 1
    grad_accum = 0
    while packing_max_batch_len > max_batch_len_per_gpu:
        grad_accum += 1
        samples_per_minibatch = effective_batch_size / grad_accum
        samples_per_gpu = samples_per_minibatch / num_gpus
        if int(avg_sample_len * samples_per_gpu) < dataset.get_lengths().max():
            raise RuntimeError(
                f"Effective batch size is too low for multipack sampling, max sample length={dataset.get_lengths().max()} and min packing length={int(avg_sample_len * samples_per_gpu)}. "
                "Switching to naive distributed sampling."
            )
        if is_padding:
            packing_max_batch_len = find_max_pack_len_with_padding(
                dataset,
                samples_per_minibatch,
                num_gpus,
                avg_sample_len,
                seed,
            )
        else:
            packing_max_batch_len = int((avg_sample_len) * samples_per_gpu)

    return packing_max_batch_len, grad_accum


## Multipack Distributed Batch Sampler
class _Bin(NamedTuple):
    """Helper named tuple for `lpt_packed_batch`"""

    fill: int  # sum of items in _Bin
    rank: int  # device rank _Bin is associated with


def lpt_packed_batch(
    lengths: np.ndarray,
    batch_max_length: int,
    num_replicas: int,
    start_index: int,
    rank: int,
) -> None | list:
    """
    Check if lengths can be distributed into n machines with at most batch_max_length per machine and return local rank's batch.

    Uses the LPT (Longest processing time first scheduling) algorithm
    Time: O(|lengths| log |lengths| + |lengths| log n)

    Returns:
    `None` if unable to find a valid packing. Otherwise return the batch indices that correspond to `rank`.
    """

    # Greedily assign lengths (in decreasing order) to the least full rank until they are all assigned or
    # we run out of space.
    local_batch = []
    heap = [_Bin(0, i) for i in range(num_replicas)]

    # sort in descending order
    indices = np.argsort(lengths)[::-1]

    for idx, size in zip(indices, lengths[indices]):
        new_fill = heap[0].fill + size
        if new_fill > batch_max_length:
            # Size doesn't fit in least full batch (or any others), can't satisfy requirements
            return None

        if heap[0].rank == rank:
            # minimum bucket corresponds to the local rank -> add idx to local batch
            local_batch.append(start_index + idx)

        _ = heapreplace(heap, _Bin(new_fill, heap[0].rank))

    return local_batch


def assign_to_packed_batches(
    lengths: np.ndarray,
    batch_max_length: int,
    rank: int,
    num_replicas: int,
) -> list[NDArray]:
    """Distribute lengths to batches across all ranks, while respecting batch_max_length. Uses a binary search + LPT algorithm

    Args:
        lengths (np.ndarray): array of dataset sample lengths
        batch_max_length (int): maximum allowed sum of lengths in batch
        rank (int): local rank to collect batches for
        num_replicas (int): world size to distribute batches to

    Returns:
        tuple[list, int, int]:
            - list of np.arrays containing the indices for each batch on the local rank
            - sum of dataset lengths included (total sum of lengths in dataset minus any that were dropped at end of dataset)
            - total token capacity if each batch maxed out batch_max_length
    """

    lengths_so_far = 0
    start_index = 0
    result = []
    lengths_cumsum = np.cumsum(lengths)

    # binary search for max integer x such that the next x elements in shuffled lengths array can be packed into `num_replicas` batches
    # Add the local rank's batch to `result` and repeat until end of dataset
    while True:
        # binary search in [1, 1 + upper bound for x)
        left = 1
        right = 1 + np.searchsorted(
            lengths_cumsum[start_index:],
            lengths_so_far + batch_max_length * num_replicas,
            "right",
        )

        batch = None
        while right - left > 1 and right > num_replicas:
            mid = (left + right) // 2
            batch = lpt_packed_batch(
                lengths[start_index : start_index + mid],
                batch_max_length,
                num_replicas,
                start_index,
                rank,
            )
            if batch is None:
                right = mid
            else:
                left = mid

        if batch is None:
            batch = lpt_packed_batch(
                lengths[start_index : start_index + left],
                batch_max_length,
                num_replicas,
                start_index,
                rank,
            )

        if left < num_replicas:
            # Can't allocate at least one length to each rank
            # Note: if left >= num_replicas, we're guaranteed to have at least one length on each machine
            # because LPT assigns values greedily to the least full machine on each step
            break

        start_index += left
        lengths_so_far = lengths_cumsum[start_index - 1]

        # append only result for local rank (already filtered in lpt_packed_batch)
        result.append(batch)

    return result


class BaseDistributedBatchSampler(Sampler):
    def __init__(
        self,
        batch_max_length: int,
        lengths: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.batch_max_length = batch_max_length
        self.lengths = np.array(lengths)
        if self.lengths.ndim != 1:
            msg = f"lengths must be coercible into a 1-dimensional numpy array. Instead found a {self.lengths.ndim}d array."
            raise ValueError(msg)

        self.valid_indices = np.nonzero(self.lengths <= self.batch_max_length)[0]
        if len(self.valid_indices) < len(self.lengths):
            # todo: change this to a warning
            print(
                f"\033[33mDropping {len(self.lengths) - len(self.valid_indices)} samples longer than batch_max_length. Ensure that the right max_batch_length is used during data processing.\033[0m"
            )

        self.last_generation = (-1, None)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        batches = self.generate_batches()
        return iter(batches)

    def __len__(self):
        return self.num_batches()

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)


class MultipackDistributedBatchSampler(BaseDistributedBatchSampler):
    def __init__(
        self,
        batch_max_length: int,
        lengths: ArrayLike,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ):
        """Efficient distributed packing sampler for linear attention style models

        Args:
            batch_max_length (int): max number of tokens in a single batch per device
            lengths (ArrayLike[int]): the lengths of each sample in the dataset
            num_replicas (int | None, optional): The number of replicas to split the dataset across. Defaults to None.
            rank (int | None, optional): The local rank to collect batches for. Defaults to None.
            seed (int, optional): Seed for RNG, must be the same on all ranks. Defaults to 0.
        """
        super().__init__(batch_max_length, lengths, num_replicas, rank, seed)

    def generate_batches(self) -> list[NDArray]:
        """Generate batches for local rank

        Returns:
            list[NDArray]: list of np.arrays containing the indices for each batch on the local rank
        """

        if self.last_generation[0] == self.epoch:
            return self.last_generation[1]

        rng = np.random.default_rng(seed=self.seed + self.epoch)
        indices = rng.permutation(self.valid_indices)

        batches = assign_to_packed_batches(
            self.lengths[indices], self.batch_max_length, self.rank, self.num_replicas
        )

        # Currently the indices in batches are relative to the shuffled self.lengths[indices]
        # We translate them so that they are instead relative to the overall unshuffled self.lengths array
        batches = [indices[batch] for batch in batches]

        # Cache result
        self.last_generation = (self.epoch, batches)
        return batches


## Padding Distributed Batch Sampler
class _HeapBucket(NamedTuple):
    # Note: Negative values are negative for sorting keys because heapq uses a min-heap
    neg_cost: int  # Negative total cost of elements in bucket being padded to max length in bucket
    neg_num_els: int  # Negative number of elements in bucket
    min_ind: int  # Index of minimum value in bucket
    max_ind: int  # Index of maximum value in bucket


def compute_bucket_bounds(
    lengths: NDArray, num_buckets: int, min_bucket_size: int
) -> NDArray:
    """Compute bucket seperators for lengths data using greedy bucket splitting algorithm

    Args:
        lengths (NDArray[int]): the lengths of dataset samples to bucket.
        num_buckets (int, optional): maximum number of buckets to produce. It is possible that less than
            `num_buckets` buckets will be produced if producing more buckets would violate the `min_bucket_size`
        min_bucket_size (int, optional): the minimum number of elements in any bucket.

    Note: this algorithm does not produce optimal bucket splits but does attempt to greedily reduce the amount of padding tokens required
    and empirically outperforms simple equidistant or equidensity bucketing approaches on real datasets.

    Returns:
        NDArray: max values (exclusive) for each bucket. shape = [n] where n <= `num_buckets`
    """

    # Define: cost of bucket C(b) = max(b) * len(b)
    # Assign all data to 1 bucket to start
    # While number of buckets < num_buckets:
    #  take most expensive bucket and split it into buckets (b_l, b_r) such that C(b_l) + C(b_r) is minimized

    lengths = np.sort(lengths)
    n_lengths = len(lengths)
    done_buckets = []

    # Note: we use a min-heap to keep track of the most expensive bucket, therefore cost and num elements (tie-breaker) are stored as negative values
    heap = [_HeapBucket(-n_lengths * lengths[-1], -n_lengths, 0, n_lengths)]

    while heap and len(done_buckets) + len(heap) < num_buckets:
        bucket = heappop(heap)

        assert -bucket.neg_num_els >= min_bucket_size
        assert bucket.max_ind - bucket.min_ind == -bucket.neg_num_els

        if -bucket.neg_num_els < min_bucket_size * 2:
            # _HeapBucket is too small to split, mark as done by removing from heap
            done_buckets.append(bucket)
            continue

        # Find optimal split index
        min_cost = float("inf")
        split_idx = -1

        # `step` makes inner loop O(1) for large datasets
        step = max(1, len(lengths) // 10000)

        # For loop is guaranteed to run at least once (by previous if condition)
        for i in range(
            bucket.min_ind + min_bucket_size, bucket.max_ind - min_bucket_size + 1, step
        ):
            n_el_left = i - bucket.min_ind
            n_el_right = bucket.max_ind - i

            cost = n_el_left * lengths[i - 1] + n_el_right * lengths[bucket.max_ind - 1]
            if cost < min_cost:
                min_cost = cost
                split_idx = i

        # Split and push new buckets onto heap
        n_el_left = split_idx - bucket.min_ind
        n_el_right = bucket.max_ind - split_idx
        cost_left = n_el_left * lengths[split_idx - 1]
        cost_right = n_el_right * lengths[bucket.max_ind - 1]
        heappush(heap, _HeapBucket(-cost_left, -n_el_left, bucket.min_ind, split_idx))
        heappush(heap, _HeapBucket(-cost_right, -n_el_right, split_idx, bucket.max_ind))

    bucket_splits = []
    for b in done_buckets + heap:
        if b.max_ind < n_lengths:
            bucket_splits.append(lengths[b.max_ind])
        else:
            # Special handling for last bucket because b.max_ind is exclusive
            bucket_splits.append(lengths[-1] + 1)

    return np.array(sorted(bucket_splits))


def assign_to_padded_batches(
    lengths: NDArray, max_tokens: int, bucket_limits: NDArray
) -> list[NDArray]:
    """Uses BucketSampler algorithm to distribute lengths to padded batches

    BucketSampler: https://aclanthology.org/2023.findings-emnlp.782.pdf

    Args:
        lengths (NDArray): The lengths of dataset samples
        max_tokens (int): maximum number of tokens (including padding) in a batch
        bucket_limits (NDArray): Max value (exclusive) for each bucket. For example
            for buckets [a, b), [b, c), [c, d), bucket_limits = [b, c, d]

    Returns:
        - list of np.arrays containing the indices for each batch
    """

    num_buckets = len(bucket_limits)
    bucket_indices = [[] for _ in range(num_buckets)]  # indices in each bucket
    bucket_max = [0 for _ in range(num_buckets)]  # max value in each bucket
    batches = []

    # Iterate through lengths, assigning to buckets
    # Once a bucket is full, it becomes a batch and the bucket is reset
    for idx, size in enumerate(lengths):
        bucket_idx = np.searchsorted(bucket_limits, size, "left")
        new_max_length = max(bucket_max[bucket_idx], size)
        new_num_elements = len(bucket_indices[bucket_idx]) + 1

        if new_num_elements * new_max_length <= max_tokens:
            bucket_max[bucket_idx] = new_max_length
            bucket_indices[bucket_idx].append(idx)
        else:
            batches.append(bucket_indices[bucket_idx])
            bucket_indices[bucket_idx] = [idx]
            bucket_max[bucket_idx] = size

    # Combine leftover buckets greedily
    new_bucket_indices = []
    new_bucket_max = 0
    for bucket_idx in range(num_buckets):
        for idx in bucket_indices[bucket_idx]:
            size = lengths[idx]
            new_max_length = max(new_bucket_max, size)
            new_num_elements = len(new_bucket_indices) + 1

            if new_num_elements * new_max_length <= max_tokens:
                new_bucket_max = new_max_length
                new_bucket_indices.append(idx)
            else:
                batches.append(new_bucket_indices)
                new_bucket_indices = [idx]
                new_bucket_max = size

    if new_bucket_indices:
        batches.append(new_bucket_indices)

    return batches


class PaddingDistributedBatchSampler(BaseDistributedBatchSampler):
    def __init__(
        self,
        batch_max_length: int,
        lengths: list[int],
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        num_buckets: int = 200,
        bucket_strategy: str = "min_split",
        min_bucket_size: int = 1,
    ):
        """Efficient distributed padding sampler for quadratic attention models

        Args:
            batch_max_length (int): max number of tokens in a single batch per device
            lengths (ArrayLike[int]): the lengths of each sample in the dataset
            num_replicas (int | None, optional): The number of replicas to split the dataset across. Defaults to None.
            rank (int | None, optional): The local rank to collect batches for. Defaults to None.
            seed (int, optional): Seed for RNG, must be the same on all ranks. Defaults to 0.
            num_buckets (int, optional): Number of buckets to split dataset into. Defaults to 200.
            bucket_strategy (str, optional): Algorithm for selecting bucket boundaries. Choice of ["min_split", "equidistant", "equidensity"]. Defaults to "min_split".
            min_bucket_size (int, optional): Minimum number of samples in each bucket. Only when bucket_strategy=="min_split". Defaults to 1.
        """
        super().__init__(batch_max_length, lengths, num_replicas, rank, seed)

        if bucket_strategy == "min_split":
            bucket_splits = compute_bucket_bounds(
                lengths, num_buckets=num_buckets, min_bucket_size=min_bucket_size
            )
        elif bucket_strategy == "equidistant":
            max_length = max(lengths)
            min_length = min(lengths)
            step = 1 / num_buckets
            bucket_splits = np.array(
                [
                    math.ceil(
                        (step + step * x) * (max_length - min_length) + min_length
                    )
                    for x in range(num_buckets)
                ]
            )
            # Make last bucket include longest element (exclusive)
            bucket_splits[-1] = lengths[-1] + 1
        elif bucket_strategy == "equidensity":
            step = 1 / num_buckets
            bucket_splits = np.quantile(
                lengths, np.array([step + step * x for x in range(num_buckets)])
            )
            # Make last bucket include longest element (exclusive)
            bucket_splits[-1] = lengths[-1] + 1
        else:
            msg = f"Invalid value for bucket_strategy: {bucket_strategy}. Must be one of {{'min_split', 'equidistant', 'equidensity'}}."
            raise ValueError(msg)

        self.bucket_splits = bucket_splits

    def generate_batches(self):
        """Generate batches for local rank

        Returns:
            list[NDArray]: list of np.arrays containing the indices for each batch on the local rank
        """

        if self.last_generation[0] == self.epoch:
            return self.last_generation[1]

        rng = np.random.default_rng(seed=self.seed + self.epoch)
        indices = rng.permutation(self.valid_indices)

        batches = assign_to_padded_batches(
            self.lengths[indices], self.batch_max_length, self.bucket_splits
        )

        # shuffle batches
        batches = [batches[i] for i in rng.permutation(len(batches))]

        # make len(batches) a multiple of num_replicas
        extra_batches = len(batches) % self.num_replicas
        if extra_batches:
            batches = batches[:-extra_batches]

        # Filter out batches that belong to other ranks
        batches = [
            batch
            for idx, batch in enumerate(batches)
            if idx % self.num_replicas == self.rank
        ]

        # Currently the indices in batches are relative to the shuffled self.lengths[indices]
        # We translate them so that they are instead relative to the overall unshuffled self.lengths array
        batches = [indices[batch] for batch in batches]

        # Cache result
        self.last_generation = (self.epoch, batches)
        return batches
