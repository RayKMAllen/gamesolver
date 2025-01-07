import torch
import random

# see https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py
# https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"

class History:

    def __init__(self, max_len=100000, per=False, eps=1e-2, alpha=0.1, beta=0.1):

        self.states = [None] * max_len
        self.actions = [None] * max_len
        self.next_states = [None] * max_len
        self.next_state_possible_actions = [None] * max_len
        self.fr_mix = [None] * max_len
        self.final_rewards = [None] * max_len
        
        
        self.max_len = max_len
        self.per = per    # Prioritized Experience Replay
        self.length = 0
        self.counter = 0

        if self.per:
            
            self.tree = SumTree(size=max_len)
            
            # PER params
            self.eps = eps  # minimal priority, prevents zero probabilities
            self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
            self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
            self.max_priority = eps  # priority for new samples, init as eps

    def update(self, state, action, next_state, nspa, fr_mix, final_reward):
        
        if self.per:
            # store transition index with maximum priority in sum tree
            self.tree.add(self.max_priority, self.counter)

        self.states[self.counter] = state
        self.actions[self.counter] = action
        self.next_states[self.counter] = next_state
        self.next_state_possible_actions[self.counter] = nspa
        self.fr_mix[self.counter] = fr_mix
        self.final_rewards[self.counter] = final_reward
        
        self.counter = (self.counter + 1) % self.max_len
        self.length = min(self.max_len, self.length + 1)

    def data_from_indexes(self, indexes):
        
        batch_states = [self.states[i] for i in indexes]
        batch_actions = [self.actions[i] for i in indexes]
        batch_next_states = [self.next_states[i] for i in indexes]
        batch_nspas = [self.next_state_possible_actions[i] for i in indexes]
        batch_fr_mix = [self.fr_mix[i] for i in indexes]
        batch_final_rewards = [self.final_rewards[i] for i in indexes]
        
        batch_states = torch.stack(batch_states)
        batch_actions = torch.tensor(batch_actions, dtype=int)
        batch_next_states = torch.stack(batch_next_states)
        batch_nspas = torch.tensor(batch_nspas)
        batch_fr_mix = torch.tensor(batch_fr_mix, dtype=torch.float32)
        batch_final_rewards = torch.tensor(batch_final_rewards, dtype=torch.float32)
        
        return batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards

    def sample_uniform(self, batch_size):
        bs = batch_size if len(self.actions) >= batch_size else len(self.actions)
        
        # choose randomly without deletion
        sample_idxs = random.sample(range(self.length), bs)
        
        batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards = self.data_from_indexes(sample_idxs)
        
        return batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards
    
    def sample_per(self, batch_size):
        bs = batch_size if len(self.actions) >= batch_size else len(self.actions)
        
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(bs, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
        
        probs = priorities / self.tree.total
        
        weights = (self.length * probs) ** -self.beta
        weights = weights / weights.max()
        
        batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards = self.data_from_indexes(sample_idxs)
        
        return batch_states, batch_actions, batch_next_states, batch_nspas, batch_fr_mix, batch_final_rewards, weights, tree_idxs
    
    def yield_batch(self, batch_size):
        
        if not self.per:
            
            return self.sample_uniform(batch_size)
        
        else:
            
            return self.sample_per(batch_size)

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)        
