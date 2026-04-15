import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, obs_dim, buffer_size, device, store_next_state=True):
        """Initialize a ReplayBuffer object.
        Arguments:
            buffer_size (int): maximum size of buffer
        """
        self.states = torch.zeros(buffer_size, obs_dim).to(device)
        self.store_next_state = store_next_state
        self.next_states = torch.zeros(buffer_size, obs_dim).to(device) if store_next_state else None
        self.buffer_size = buffer_size
        self.device = device

        self.step = 0
        self.num_samples = 0
    
    def insert(self, states, next_states=None):
        """Add new states to memory."""
        if self.store_next_state and next_states is None:
            raise ValueError("ReplayBuffer expects `next_states` when `store_next_state=True`.")

        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states
        if end_idx > self.buffer_size:
            self.states[self.step:self.buffer_size] = states[:self.buffer_size - self.step]
            if self.store_next_state:
                self.next_states[self.step:self.buffer_size] = next_states[:self.buffer_size - self.step]
            self.states[:end_idx - self.buffer_size] = states[self.buffer_size - self.step:]
            if self.store_next_state:
                self.next_states[:end_idx - self.buffer_size] = next_states[self.buffer_size - self.step:]
        else:
            self.states[start_idx:end_idx] = states
            if self.store_next_state:
                self.next_states[start_idx:end_idx] = next_states

        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            if self.store_next_state:
                yield (
                    self.states[sample_idxs].to(self.device),
                    self.next_states[sample_idxs].to(self.device),
                )
            else:
                yield self.states[sample_idxs].to(self.device)
