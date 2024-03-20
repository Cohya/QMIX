import numpy as np 


        
# replay buffer
class SumTree:
    # little modified from https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    def __init__(self, capacity):
        self.capacity = capacity    # N, the size of replay buffer, so as to the number of sum tree's leaves
        self.tree = np.zeros(2 * capacity - 1)  # equation, to calculate the number of nodes in a sum tree
        self.transitions = np.empty(capacity, dtype=object)
        self.next_idx = 0
        
        
    @property
    def total_p(self):
        return self.tree[0]

    def add(self, priority, transition):
        idx = self.next_idx + self.capacity - 1
        self.transitions[self.next_idx] = transition
        self.update(idx, priority)
        self.next_idx = (self.next_idx + 1) % self.capacity
        # print("next_idx:", self.next_idx )
        
    def update(self, idx, priority):
        change = priority - self.tree[idx]
        # print("idx:", idx) ## capacity -1 -->2*capacity -1 
        # input("d:")
        self.tree[idx] = priority
        ## Propagate to the parents
        self._propagate(idx, change)    # O(logn)
        # print("tree:", self.tree)
        
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change) ## This is updating up to the parent 

    def get_leaf(self, s, max_location):
        idx = self._retrieve(0, s, max_location)   # from root
        trans_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.transitions[trans_idx]

    # def _retrieve_old(self, idx, s):
    #     left = 2 * idx + 1 # 1
    #     right = left + 1 # 2
    #     # print("left:", left, "right:", right)
        
    #     if left >= len(self.tree):
    #         # print("out")
    #         return idx
    #     # print(self.tree[left])
    #     if s <= self.tree[left]:
    #         # print("c1")
    #         return self._retrieve(left, s)
    #     else:
    #         return self._retrieve(right, s - self.tree[left])
        
    def _retrieve(self, idx, s, max_location):
        # s == segment
        parent_index = idx

        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree) or  left_child_index >= self.capacity + max_location:
                leaf_index = parent_index
                break 
            
            if s <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                s = s - self.tree[left_child_index]
                parent_index = right_child_index
                
        return leaf_index
    
    #Recursive approach 
    def _retrieve_recursive(self, idx, s, max_location):
        left = 2 * idx + 1 # 1
        right = left + 1 # 2
        if left >= len(self.tree) or  left >= self.capacity + max_location:
            # print("out")
            return idx
        # print(self.tree[left])
        # Always look for a higher priority node
        if s <= self.tree[left]:
            # print("c1")
            return self._retrieve(left, s, max_location)
        else:
            return self._retrieve(right, s - self.tree[left], max_location)
        
class ProportionalPriorizationReplayBuffer(SumTree):
    def __init__(self,capacity, obs_dims,  alpha=0.4,
                 beta=0.4, beta_increment_per_sample = 0.001,
                 batch_size = 32,
                 agent_num = 2):
        super().__init__(capacity)
        self.p1 = 1 
        self.num_in_buffer = 0                      # total number of transitions stored in buffer
        self.beta = beta                            # # importance sampling parameter, beta=[0, 0.4, 0.5, 0.6, 1]
        self.beta_increment_per_sample = beta_increment_per_sample
        
        self.batch_size  = batch_size
        self.b_obs = np.empty((self.batch_size,obs_dims))
        self.b_actions = np.empty((self.batch_size, 2))
        self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
        self.b_next_states = np.empty((self.batch_size, obs_dims)) 
        self.b_dones = np.empty(self.batch_size, dtype= bool)
        self.margin = 0.01                          # pi = |td_error| + margin
        
        self.abs_error_upper = 1
        self.alpha = alpha                          # priority parameter, alpha=[0, 0.4, 0.5, 0.6, 0.7, 0.8] 
        self.number_of_transition_stored = 0
        
        self.agent_num = agent_num
        
    def store_transition(self, priority,
                         state, obs, action, reward, next_state, next_obs, done):
        transition = [state, obs, action, reward, next_state, next_obs, done]
        self.add(priority, transition)
        # print("priority:", priority)
        self.num_in_buffer = min(self.num_in_buffer + 1, self.capacity) 
        self.number_of_transition_stored += 1
        # print("self.num_in_buffer:", self.num_in_buffer,  self.capacity)
        
    def get_max_p(self):
        if self.num_in_buffer >= self.capacity: #was 1000
            p_max = np.max(self.tree[-self.capacity:])
            
        else:
            # index_i = max(1, self.num_in_buffer)
            # print(len(self.tree[-self.capacity:(-self.capacity + self.num_in_buffer)]))
            p_max = np.max(self.tree[-self.capacity:(-self.capacity + self.num_in_buffer)])
            # print("p_max:", p_max)
        return p_max
    
    
    ## Now we can create sample fucntion, which will be used to pick batch from our tree memory during trainign 
    # First: we sample a minibatch of the size batch_size, the range [0,total_p] into priority ranges
    # Then a value is uniformly sampled from each range
    # Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sum_tree_sample(self):
        idxes = []
        is_weights = np.empty((self.batch_size, 1))
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        if self.num_in_buffer >= self.capacity: # was 1000: 
            min_val = np.min(self.tree[-self.capacity:])
            min_prob = min_val / self.total_p
        else:
            
            min_val = np.min(self.tree[-self.capacity:(-self.capacity+ self.num_in_buffer)])
            min_prob = min_val / self.total_p


        
        max_weight = np.power(self.capacity * min_prob, -self.beta) # This is for stability 
        
        ## Calculate priority segment 
        segment = self.total_p / self.batch_size ## To create balance to the mini-batch 
        ## segment mimic the cdf of the distribution : it is a linear approximation! 
        # Sorting by p_i and sample is not efficient so we do segment style O(nlog(N))
        # segment is 0(log(N))
        state_batch = []
        observation_batch = [[] for _ in range(self.agent_num)]
        action_batch = []
        reward_batch = []
        next_state_batch = []
        next_observation_batch = [[] for _ in range(self.agent_num)]
        terminal_batch = []
        
        for i in range(self.batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1)) ## <--this will give us different level of priority TD 
            # experience that correspond to each value is retrieved!
            idx, p, t = self.get_leaf(s, max_location = self.num_in_buffer)
            idxes.append(idx)# Store to update 
            
            # self.b_obs[i], self.b_actions[i], self.b_rewards[i], self.b_next_states[i], self.b_dones[i] = t ##<--replace to batch 
            state, obs, action, reward, next_state, next_obs, done= t 
            # [state, obs, action, reward, next_state, next_obs, done]
            # P(j)
            for i in range(self.agent_num):
                observation_batch[i].append(obs[i])
                next_observation_batch[i].append(next_obs[i])
                
            state_batch.append(state)
            # observation_batch.append(obs)
            action_batch.append(action)
            reward_batch.append(reward)
            # next_observation_batch.append(next_obs)
            next_state_batch.append(next_state)
            terminal_batch.append(done)
            
            sampling_probabilities = p / self.total_p     # where p = p ** self.alpha
            is_weights[i, 0] = np.power(self.capacity * sampling_probabilities, -self.beta) / max_weight # Fixing the bias
            
        state_batch = np.array(state_batch)
        observation_batch = np.array(observation_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_observation_batch = np.array(next_observation_batch)
        next_state_batch = np.array(next_state_batch)
        terminal_batch = np.array(terminal_batch)
        return  state_batch, observation_batch, action_batch, reward_batch, next_state_batch, next_observation_batch, terminal_batch, \
                idxes, is_weights
    

    
    def get_minibatch(self):
        assert  self.batch_size < self.num_in_buffer, "Your replay buffer has lower values stored than in the required Batch size!"
        
        (state_batch, observation_batch, action_batch, reward_batch, next_state_batch, next_observation_batch, terminal_batch,
                idxes, is_weights) = self.sum_tree_sample()
        
        return state_batch, observation_batch, action_batch, reward_batch, next_state_batch, next_observation_batch, \
                terminal_batch,\
                idxes, is_weights
    
    def update_priorization(self, idxes, abs_delta):
        abs_delta += self.margin
        clipped_error = np.where(abs_delta < self.abs_error_upper, abs_delta, self.abs_error_upper) # do not agree to take high TD error 
       
        ps = np.power(clipped_error, self.alpha)
        
        for idx, p in zip(idxes, ps):## The priority is already in p**alpha 
             self.update(idx, p)
             
         
 ## In this format you should wait at leat for a full RMB capacity !! you 
## can not start earlier, the above version gives you the oportonity to start even with 
## number of samples that are at list at the size of the batch size !!                
class ProportionalPriorizationReplayBuffer_old(SumTree):
    def __init__(self,capacity, obs_dims,  alpha=0.4, beta=0.4, beta_increment_per_sample = 0.001,
                 batch_size = 32, agent_num = 2):
        super().__init__(capacity)
        self.p1 = 1 
        self.num_in_buffer = 0                      # total number of transitions stored in buffer
        self.beta = beta                            # # importance sampling parameter, beta=[0, 0.4, 0.5, 0.6, 1]
        self.beta_increment_per_sample = beta_increment_per_sample
        
        self.batch_size  = batch_size
        self.b_obs = np.empty((self.batch_size,obs_dims))
        self.b_actions = np.empty(self.batch_size, dtype=np.int8)
        self.b_rewards = np.empty(self.batch_size, dtype=np.float32)
        self.b_next_states = np.empty((self.batch_size, obs_dims)) 
        self.b_dones = np.empty(self.batch_size, dtype= bool)
        self.margin = 0.01                          # pi = |td_error| + margin
        
        self.abs_error_upper = 1
        self.alpha = alpha                          # priority parameter, alpha=[0, 0.4, 0.5, 0.6, 0.7, 0.8] 
        self.number_of_transition_stored = 0
        self.agent_num = agent_num
        
    def store_transition(self, priority, obs, action, reward, next_state, done):
        transition = [obs, action, reward, next_state, done]
        self.add(priority, transition)
        self.num_in_buffer = min(self.num_in_buffer + 1, self.capacity) 
        self.number_of_transition_stored += 1
        # print("self.num_in_buffer:", self.num_in_buffer,  self.capacity)
        

    
    def get_max_p(self):
        p_max = np.max(self.tree[-self.capacity:])
        return p_max
    
    
    # proportional prioritization sampling
    def sum_tree_sample_n(self):
        idxes = []
        is_weights = np.empty((self.batch_size, 1))
        self.beta = min(1., self.beta + self.beta_increment_per_sample)
        # calculate max_weight
        min_prob = np.min(self.tree[-self.capacity:]) / self.total_p
        # max_prob = np.max(self.tree[-self.capacity:]) / self.total_p
        
        max_weight = np.power(self.capacity * min_prob, -self.beta) # This is for stability 
        segment = self.total_p / self.batch_size ## To create balance to the mini-batch 

        
        for i in range(self.batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1)) ## <--this will give us different level of priority TD 
            idx, p, t = self.get_leaf(s)
            idxes.append(idx)# Store to update 
            self.b_obs[i], self.b_actions[i], self.b_rewards[i], self.b_next_states[i], self.b_dones[i] = t ##<--replace to batch 
            # P(j)
            sampling_probabilities = p / self.total_p     # where p = p ** self.alpha
            is_weights[i, 0] = np.power(self.capacity * sampling_probabilities, -self.beta) / max_weight # Fixing the bias
        return idxes, is_weights
    
    
    def get_minibatch(self):
        assert  self.batch_size < self.num_in_buffer, "Your replay buffer has lower values stored than in the required Batch size!"
        
        idxes, is_weights = self.sum_tree_sample()
        
        return self.b_obs, self.b_actions, self.b_rewards, self.b_next_states, self.b_dones, idxes, is_weights
    
    def update_priorization(self, idxes, abs_delta):
        abs_delta += self.margin
        clipped_error = np.where(abs_delta < self.abs_error_upper, abs_delta, self.abs_error_upper) # do not agree to take high TD error 
       
        ps = np.power(clipped_error, self.alpha)
        
        for idx, p in zip(idxes, ps):## The priority is already in p**alpha 
             self.update(idx, p)    
        
        