
import tensorflow as tf 
import copy
import numpy as np 
import time

class MixingNetwork(tf.keras.Model):
    def __init__(self, agents_nets, embed_shape):
        
        super().__init__(MixingNetwork)
        self.agent_nets = agents_nets
        self.number_of_agents = len(agents_nets)
        self.embed_shape = embed_shape
        self.history  = agents_nets[0].input_shape[0] # was1
        self.agent_output_dim = agents_nets[0].output_shape # action_dims
        
        
        ### Bulding the layers 
        self.hyper_w1_1 = tf.keras.layers.Dense(embed_shape,
                                                activation='relu', 
                                                use_bias=True)
        
        self.hyper_w1_2 = tf.keras.layers.Dense(embed_shape * 
                                                self.number_of_agents *
                                                self.agent_output_dim,
                                                activation='relu', 
                                                use_bias = True)
        
        self.hyper_b1 = tf.keras.layers.Dense(self.embed_shape) 
        
        self.hyper_w2_1 = tf.keras.layers.Dense(
            self.embed_shape, activation='relu', use_bias=True)
        
        self.hyper_w2_2 = tf.keras.layers.Dense(
            self.embed_shape, activation='linear', use_bias=True)
        
        self.hyper_b2 = tf.keras.layers.Dense(1, activation="relu")
        
        
        
    def call(self,inputs):
        agents_inputs = inputs[0]
        states = inputs[1]
        masks = inputs[2]
        batch_size = states.shape[0]
        
        agents_outputs = []
        
        for agent_net, agent_input, mask in zip(self.agent_nets,
                                                agents_inputs,
                                                masks):
            agent_out = agent_net(agent_input)
            agent_out = tf.multiply(agent_out, mask) # this is just a dot product 
            agents_outputs.append(agent_out)
            #print(agent_out.shape) # (1,2)
            #input("holdoing")
            
        xx = self.hyper_w1_1(states)
        w1 = tf.abs(self.hyper_w1_2(xx))
        print("xx:", xx.shape)
        print("w1:", w1.shape)
        
        agents_outputs = tf.concat(agents_outputs, 1)
        print("agents_outputs:", agents_outputs.shape) #(1,4)  
        #input("holding")
        agents_outputs = tf.expand_dims(agents_outputs, 1)
        #print(agents_outputs.shape)#(1,1,4)
        #input("holding")
        
        w1 = tf.reshape(w1, [
            batch_size, self.agent_output_dim * self.number_of_agents, -1])
        print("w1:", w1.shape)
       # print(w1.shape) #(1, 4, 60 ) 60 == embedding ! 
       # print( batch_size, self.agent_output_dim ,self.number_of_agents)
        #input("holding")
        b1 = self.hyper_b1(states)
        
        b1 = tf.reshape(b1, [batch_size, 1, -1]) # -1 take automatically the size needed
        
        ggg =  tf.matmul(agents_outputs, w1) 
        print("gg:", ggg.shape)
        hidden = tf.keras.activations.elu(
            tf.matmul(agents_outputs, w1) + b1)
        
        print("hidden:", hidden.shape)
        w2 = tf.abs(self.hyper_w2_2(self.hyper_w2_1(states)))
        
        w2 = tf.reshape(w2, [batch_size, self.embed_shape, 1])
        b2 = self.hyper_b2(states)
        b2 = tf.reshape(b2, [batch_size, 1, 1])
        y = tf.matmul(hidden, w2) + b2
        
        q_tot = tf.reshape(y, [-1, 1])
        print(q_tot.shape, batch_size)
        print(states.shape)
        # input("hold")
        
        return q_tot
    
class QMIX(object):
    def  __init__(self, 
                  agents,
                  replay_memory,
                  experience_replay_buffer,
                  gamma = 0.99,
                  batch_size = 32,
                  loss_fn = tf.keras.losses.MeanSquaredError(),
                  optimizer = tf.keras.optimizers.RMSprop(),
                  update_interval = 200,
                  embed_shape = 60, 
                  lr = 0.0005,
                  agent_action_num = 2,
                  use_double_dqb = False,
                  use_priorized_replay = False,
                  share_weights = False):
        
        self.activaeteee = 0
        self.agents = agents 
        self.replay_memory = replay_memory
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.experience_replay_buffer = experience_replay_buffer
        self.use_priorized_replay = use_priorized_replay
        self.step = 0
        self.train_interval = 1
        self.warmup_steps = 60
        
        self.prev_state = None
        self.prev_observations = None
        self.agent_action_num = agent_action_num #!!!
        
        self.activate_ddqn = use_double_dqb
        ## Collect all the agents networks
        models = []
        target_models = []
        
        for agent in agents:
            models.append(agent.model)
            target_models.append(agent.target_model)
            
        self.trainable_variables = [] 
        self.target_trainable_variables = []
        self.share_weights = share_weights
        
        if self.share_weights:
            print("Activate share weights...")
            time.sleep(1)
            agent = agents[0]
            self._share_weights_fnc(update_target= True, verbose=True)
            self.trainable_variables += agent.model.trainable_variables
            self.target_trainable_variables += agent.target_model.trainable_variables
        else:
            for agent in agents:
                self.trainable_variables += agent.model.trainable_variables
                self.target_trainable_variables += agent.target_model.trainable_variables

        self.model_mixmax =  MixingNetwork(agents_nets = models,
                                           embed_shape = embed_shape)
        
        self.target_model_mixmax = MixingNetwork(agents_nets = target_models,
                                           embed_shape = embed_shape)
        
        ### activaet the mixmax 
        
        # self.trainable_variables + self.model_mixmax.trainable_variables
        
        # self.target_trainable_variables += self.target_model_mixmax.trainable_variables

        self.loss_fn = loss_fn #tf.keras.losses.MSE #loss_fn
        self.optimizer = optimizer
        
    def save(self, priority, state, observations, actions, reward, is_terminal):
        if self.prev_state is None:
            self.prev_state = copy.deepcopy(state)
            self.prev_observations = observations
            
        if self.use_priorized_replay:
            self.experience_replay_buffer.store_transition(priority, 
                                      self.prev_state, # state
                                      self.prev_observations,# obs
                                      actions, # <- save as one hot 
                                      reward,
                                      state, # <- next state 
                                      observations,#< next obs
                                      is_terminal)
        else:
            self.replay_memory.append(self.prev_state, # state
                               self.prev_observations,# obs
                               actions, # <- save as one hot 
                               reward,
                               state, # <- next state 
                               observations,#< next obs
                               terminal=is_terminal)
            

        
        self.prev_state = copy.deepcopy(state)
        self.prev_observations = copy.deepcopy(observations)
        self.step += 1 
        
    def train(self):
        loss_i, idxes, abs_delta = self._loss()
        return loss_i, idxes, abs_delta

    def _loss(self):
        loss = 0
        if self.step > self.warmup_steps \
                and self.step % self.train_interval == 0:
                    
            if self.use_priorized_replay:
                (states, observations, actions, rewards, next_states, next_observations, terminals,
                        idxes, weights) = self.experience_replay_buffer.get_minibatch() 
            else:
                states, observations, actions, rewards, next_states, next_observations, terminals = self.replay_memory.sample(
                    self.batch_size)
                idxes = None
                weights = None
                
            

            print("state shape:",states.shape)
            input("hold")
            # print(states.shape,states1.shape, '\n', # not good 
            #       observations.shape,observations1.shape,'\n',
            #       actions.shape, actions1.shape, '\n',
            #       rewards.shape, rewards1.shape, '\n',
            #       next_states.shape,next_states1.shape,'\n',
            #       next_observations.shape,next_observations1.shape,'\n', # not good 
            #       terminals.shape,terminals1.shape)
            
            # input("holds")
            
            # actions --> samplesx agents_num x actions_num
            rewards = np.array(rewards).reshape(-1, 1)
            terminals = np.array(terminals).reshape(-1, 1)
            next_observations = np.array(next_observations)
            next_states = np.array(next_states)

            masks, target_masks = [], []
            for idx, (agent, next_observation) in enumerate(
                    zip(self.agents, next_observations)):
                ## if you want to use double DQN you should change it 
                if self.activate_ddqn:
                    agent_out = agent.model(next_observation) 
                else:
                    agent_out = agent.target_model(next_observation)
                
                argmax_actions = tf.keras.backend.argmax(agent_out)
                target_mask = tf.one_hot(
                    argmax_actions, depth=self.agent_action_num)
                target_masks.append(target_mask)
                masks.append(actions[:, idx, :])
            # print(actions)
            # print(masks)
            masks = tf.convert_to_tensor(masks)
            target_masks = tf.convert_to_tensor(target_masks)

            target_q_values = self._predict_on_batch(
                next_states, next_observations, target_masks, self.target_model_mixmax)
            
            
            discounted_reward_batch = self.gamma * target_q_values * terminals
            targets = rewards + discounted_reward_batch

            # Set up logging.
            # from  datetime import datetime
            # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            # logdir = 'logs/func/%s' % stamp
            # writer = tf.summary.create_file_writer(logdir)
            # tf.summary.trace_on(graph=True, profiler=True)

            observations = np.array(observations)
            states = np.array(states)
            states = tf.convert_to_tensor(states, dtype=tf.float32)
            observations = tf.convert_to_tensor(
                observations, dtype=tf.float32)
            
            if self.use_priorized_replay:
                loss, abs_delta = self._train_on_batch(
                    states, observations, masks, targets, weights)
            else:
                loss, abs_delta = self._train_on_batch(
                    states, observations, masks, targets, None)
            # with writer.as_default():
            #     tf.summary.trace_export(
            #             name="my_func_trace",
            #             step=0,
            #             profiler_outdir=logdir)

        if self.update_interval > 1:
            # hard update
            self._hard_update_target_model()
        else:
            # soft update
            self._soft_update_target_model()
        self.step += 1
        
        # print(abs_delta)
        return loss, idxes, abs_delta

    def _predict_on_batch(
            self,
            states,
            observations,
            masks,
            model):
        q_values = model([observations, states, masks])
        return q_values ## thsi is Q_total
    
    
    def _compute_q_values(self, state):
        q_values = self.target_model.predict(np.array([state]))
        return q_values[0]

    # @tf.function
    def _train_on_batch(self, states, observations, masks, targets,weights):
        targets = tf.stop_gradient(targets)
        with tf.GradientTape() as tape:
            tape.watch(observations)
            tape.watch(states)
            y_preds = self.model_mixmax([observations, states, masks])
            ######### or should I do it mself the loss  of MSE
            # loss_value = self.loss_fn(targets, y_preds,  sample_weight = weights)
            if self.use_priorized_replay:
                a = tf.expand_dims(self.loss_fn(y_true = targets , y_pred = y_preds), axis = 1)
                k =  tf.math.multiply(a, weights)
                # print(k.shape)
                loss_value = tf.reduce_mean(k)*0.5
                
                
            else:
                loss_value = tf.reduce_mean(
                                       self.loss_fn(y_true = targets , y_pred = y_preds)
                                       )*0.5
                
        self.last_q_values = y_preds  # @todo
        self.last_targets = targets  # @todo
        
        #!!!@ Move the adding of the trainable params to the constructor! 
        self.activaeteee += 1
        if self.activaeteee == 1:
            self.trainable_variables += self.model_mixmax.trainable_variables
        grads = tape.gradient(loss_value, self.trainable_variables)
        #########Clip##########
        
        self.optimizer.apply_gradients(
            zip(grads, self.trainable_variables))
        
        if self.share_weights:
            self._share_weights_fnc(update_target= False, verbose=False)
            
        abs_delta = np.abs(targets - y_preds)
        
        return loss_value.numpy(),abs_delta
    

        
    def _share_weights_fnc(self, update_target = False, verbose = False):
        for i, agent in enumerate(self.agents):
            if i  == 0 :
                w = agent.get_weights()
                if update_target:
                    agent._hard_update_target_model()
            else:
                agent.load_weights(w)
                if update_target:
                    agent._hard_update_target_model()
                
        if verbose:
            print("All agents set with the same weights!")
            

    def _hard_update_target_model(self):
        """ for hard update """
        if self.step % self.update_interval == 0:
            self.target_model.set_weights(self.model.get_weights())
        for agent in self.agents:
            agent._hard_update_target_model()

    def _soft_update_target_model(self):
        target_model_weights = np.array(self.target_model.get_weights())
        model_weights = np.array(self.model.get_weights())
        new_weight = (1. - self.update_interval) * target_model_weights \
            + self.update_interval * model_weights
        self.target_model.set_weights(new_weight)
        for agent in self.agents:
            agent._soft_update_target_model()
            
            
    def get_qmix_output(self):
        """
            for debug
        """
        obs = np.array([[[[0., 0., 1.]]], [[[0., 0., 1.]]]])
        st = np.array([[0., 0., 1.]])
        mk = np.array([[[1., 0.]], [[1., 0.]]])

        obs = tf.convert_to_tensor(obs, dtype=np.float32)
        st = tf.convert_to_tensor(st, dtype=np.float32)
        mk = tf.convert_to_tensor(mk, dtype=np.float32)

        result = {}
        result[(0, 0)] = round(self.model_mixmax([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[1., 0.]], [[0., 1.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(0, 1)] = round(self.model_mixmax([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[0., 1.]], [[1., 0.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(1, 0)] = round(self.model_mixmax([obs, st, mk]).numpy()[0][0], 2)

        mk = np.array([[[0., 1.]], [[0., 1.]]])
        mk = tf.convert_to_tensor(mk, dtype=np.float32)
        result[(1, 1)] = round(self.model_mixmax([obs, st, mk]).numpy()[0][0], 2)

        return result      
            