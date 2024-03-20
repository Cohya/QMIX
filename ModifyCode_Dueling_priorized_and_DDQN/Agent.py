
import tensorflow as tf 
from collections import deque
import numpy as np 

class Agent():
    def __init__(self, model, target_model, i_d, history = 1):
        
        self.model = model 
        self.target_model = target_model 
        self.i_d = i_d
        self.history = history

        self.trajectory = None 
        self.last_q_values = None 
        
        
    def _init_deque(self, obs):
        trajectory = deque(maxlen = self.history)
        for _ in range(self.history):
            trajectory.append(obs)
        return trajectory
    
    def reset(self, obs):
        self.trajectory = self._init_deque(obs)
        self.obs = obs
        self.prev_obs = obs 
        
     
    def observe(self, obs):
        self.prev_obs = self.obs
        self.obs = obs
        self.trajectory.append(obs)
    
    def compute_q_a_values_based_main_net(self, state, model):
        inputs = tf.convert_to_tensor(list(state))
        
        inputs = tf.expand_dims(inputs, 0) # give it the extradimention to say that it is a one sample 
        # print(' ')
        # print(inputs.shape)
        # print(inputs)
        # input("sdf")
        q_s_a = self.model(inputs)
        
        # print(q_s_a)
        # input("hallo:")
        return q_s_a[0]
      
      
    def get_action(self, eps = 0):
        ## Make sure that you let the agent observe the obs before action 
        
        q_values = self.compute_q_a_values_based_main_net(self.trajectory, 
                                                          self.model)
        self.last_q_values = q_values
        
        assert q_values.ndim == 1, "Check the dims of q_values"
        
        if np.random.uniform() < eps:
            action  = np.random.randint(0,len(q_values))
            
        else:
            action = tf.argmax(q_values)
            
        return action 
    
    
    def _hard_update_target_model(self):
        """ for hard update """
        self.target_model.model.set_weights(self.model.model.get_weights())

    def _soft_update_target_model(self):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)
    
    def load_weights(self,w):
        for w_i, w_local in zip(w, self.model.trainable_variables):
            w_local.assign(w_i)
    
    def get_weights(self):
        weights = []
        
        for w in self.model.trainable_variables:
            weights.append(w.numpy())
        
        return weights
        
        
        
