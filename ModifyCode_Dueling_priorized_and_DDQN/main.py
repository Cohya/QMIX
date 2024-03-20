
from two_step_env import TwoStepEnv
import copy 
import numpy as np 
import matplotlib.pyplot as plt
from  Keras_network import ANNasClass, ANN,ANNasClassDueling
from Agent import Agent
from replay_memory import RandomMemory
from ProportionalPriorizationReplayBuffer import ProportionalPriorizationReplayBuffer
from  Qmix import QMIX
import tensorflow as tf 
import tqdm
num_of_agents= 2


env = TwoStepEnv()

number_of_actions = len(env.action_space) # 2-->[0,1]

observation_dims = len(env.observation_space) # 3 -->[0,1,2]

history = 1 # trajectory_len
input_shape = (history, observation_dims)

### Lets create the agents 
agents = []

for i in range(num_of_agents):
    model = ANNasClassDueling(input_shape = input_shape,
                output_shape = number_of_actions)
    
    target_model = ANNasClassDueling(input_shape = input_shape,
                output_shape = number_of_actions)
    
    agent = Agent(model = model,
                  target_model = target_model,
                  i_d = i,
                  history = history)
    
    # fix target with main network weights
    agent.target_model.model.set_weights(model.model.get_weights())
    
    
    agents.append(agent)
                  
         
### Create the QMIX net 
loss_fn = tf.keras.losses.MSE
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
batch_size = 32
prm = ProportionalPriorizationReplayBuffer(capacity = 500, obs_dims = observation_dims)
memory = RandomMemory(limit=500)
use_priorized_replay = False
qmix = QMIX(
    agents=agents,
    replay_memory=memory,
    experience_replay_buffer = prm,
    batch_size=batch_size,
    loss_fn=loss_fn,
    optimizer=optimizer,
    use_double_dqb = True,
    use_priorized_replay = use_priorized_replay,
    share_weights = True)

## Make all agent woth the same weights
# qmix.share_weights(verbose=True)


episode_reward_history = []
loss_history = []
episode_reward_mean = 0
loss_mean = 0

max_episodes = 1000
# s = env.reset()
# init_state = tf.one_hot(s,observation_dims )
eps = 1
global_iters = 0
with tqdm.trange(max_episodes) as t:
    for episode in t:

        s = env.reset() # s shoudl be per agent 
        # init_state = [tf.one_hot(i,observation_dims) for i in s]
        for agent, s0 in zip(agents,s):
            s_i = tf.one_hot(s0,observation_dims)
            agent.reset(s_i)
        rewards = []
        for step in range(3): # <- max lingth of the game 
            actions = []
            for agent in agents:
                action = agent.get_action(eps  = eps)
                actions.append(action)
            state, reward, done = env.step(actions)
            # state_one_hot = [tf.one_hot(i, observation_dims) for i in state]
            rewards.append(reward)
            #!!!@Doto : change the logic of the game make the agent 
            # get the observation in the get action and delete the 
            # observ fucntion of it make the history out of the agent 
            trajectories = []
            
            for agent,i in zip(agents,state):
                s_i = tf.one_hot(i, observation_dims)
                agent.observe(s_i) # The
                trajectory = copy.deepcopy(agent.trajectory)
                trajectories.append(trajectory)

            one_hot_actions = []
            for action in actions:
                action = tf.one_hot(action, depth=number_of_actions)
                one_hot_actions.append(action)
          
            if use_priorized_replay:
                if global_iters == 0:
                    p = qmix.experience_replay_buffer.p1
                else:
                    p = qmix.experience_replay_buffer.get_max_p() #np.max(agent.replay_buffer.tree[-self.replay_buffer.capacity:]) 
                    
            else:
                p = None
            
            state = s_i
   
            qmix.save(p,state, trajectories, one_hot_actions, reward, done)
            # prm.store_transition([state, trajectories, one_hot_actions, reward, done])
            
            if episode > batch_size:
                loss,  idxes, abs_delta = qmix.train()
                loss_history.append(loss)
                if use_priorized_replay:
                    qmix.experience_replay_buffer.update_priorization(idxes, abs_delta)
                
            if done:
                break
            global_iters += 1
        episode_reward = np.sum(rewards)
        episode_reward_history.append(episode_reward)
        episode_reward_mean = 0.01 * episode_reward + 0.99 * episode_reward_mean
        # eps = 0.99*eps
        t.set_description(
            f"Episode:{episode},state:{env.prev_state},qmix:{qmix.get_qmix_output()}, reward:{episode_reward}")
        t.set_postfix(episode_reward_mean=episode_reward_mean)

fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
axL.plot(
    np.arange(
        len(episode_reward_history)),
    episode_reward_history,
    label="episode_reward")
axL.set_xlabel('episode')
axL.set_title("episode reward history")

axR.plot(np.arange(len(loss_history)), loss_history, label="loss")
axR.set_title("qmix's loss history")

axR.legend()
axL.legend()
plt.savefig("result2.png")