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
from heapq import heappop, heappush, heapreplace
from typing import NamedTuple
import math

# Third Party
from numpy.typing import ArrayLike, NDArray
from torch.utils.data import Sampler
import numpy as np
import torch
import torch.distributed as dist


def find_packing_max_batch_len_and_grad_accum(
    num_gpus, avg_sample_len, effective_batch_size, max_batch_len_per_gpu, dataset, seed
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

        packing_max_batch_len = int((avg_sample_len) * samples_per_gpu)

    return packing_max_batch_len, grad_accum


## Multipack Distributed Batch Sampler
class _Bin(NamedTuple):
    """Helper named tuple for `lpt_packed_batch`"""

    fill: int  # sum of items in _Bin
    rank: int  # device rank _Bin is associated with


def lpt_packed_batch(
    lengths: np.ndarray, max_len: int, replicas: int, start_index: int, rank: int
) -> None | list:
    """
    Check if lengths can be distributed into `replicas` machines with at most `max_len` tokens per machine and return local rank's batch.

    Uses the LPT (Longest processing time first scheduling) algorithm
    Time: O(|lengths| log |lengths| + |lengths| log n)

    Returns:
    `None` if unable to find a valid packing. Otherwise return the batch indices that correspond to `rank`.
    """

    # Greedily assign lengths (in decreasing order) to the least full rank until they are all assigned or
    # we run out of space.
    local_batch = []
    heap = [_Bin(0, i) for i in range(replicas)]

    # sort in descending order
    indices = np.argsort(lengths)[::-1]

    for idx, size in zip(indices, lengths[indices]):
        new_fill = heap[0].fill + size
        if new_fill > max_len:
            # Size doesn't fit in least full batch (or any others), can't satisfy requirements
            return None

        if heap[0].rank == rank:
            # minimum bucket corresponds to the local rank -> add idx to local batch
            local_batch.append(start_index + idx)

        _ = heapreplace(heap, _Bin(new_fill, heap[0].rank))

    return local_batch


def assign_to_packed_batches(
    lengths: np.ndarray, max_len: int, rank: int, replicas: int
) -> list[NDArray]:
    """Distribute lengths to batches across all ranks, while respecting batch_max_length. Uses a binary search + LPT algorithm

    Args:
        lengths (np.ndarray): array of dataset sample lengths
        max_len (int): maximum allowed sum of lengths in batch
        rank (int): local rank to collect batches for
        replicas (int): world size to distribute batches to

    Returns:
        tuple[list, int, int]:
            - list of np.arrays containing the indices for each batch on the local rank
            - sum of dataset lengths included (total sum of lengths in dataset minus any that were dropped at end of dataset)
            - total token capacity if each batch maxed out batch_max_length
    """

    lengths_so_far = 0
    ind = 0
    result = []
    lengths_cumsum = np.cumsum(lengths)

    # binary search for max integer x such that the next x elements in shuffled lengths array can be packed into `num_replicas` batches
    # Add the local rank's batch to `result` and repeat until end of dataset
    while True:
        # binary search in [1, 1 + upper bound for x)
        left = 1
        right = 1 + np.searchsorted(
            lengths_cumsum[ind:], lengths_so_far + max_len * replicas, "right"
        )

        batch = None
        while right - left > 1 and right > replicas:
            mid = (left + right) // 2
            batch = lpt_packed_batch(
                lengths[ind : ind + mid], max_len, replicas, ind, rank
            )
            if batch is None:
                right = mid
            else:
                left = mid

        if batch is None:
            batch = lpt_packed_batch(
                lengths[ind : ind + left], max_len, replicas, ind, rank
            )

        if left < replicas:
            # Can't allocate at least one length to each rank
            # Note: if left >= num_replicas, we're guaranteed to have at least one length on each machine
            # because LPT assigns values greedily to the least full machine on each step
            break

        ind += left
        lengths_so_far = lengths_cumsum[ind - 1]

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


class MultipackDistributedBatchSamplerV2(BaseDistributedBatchSampler):
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
