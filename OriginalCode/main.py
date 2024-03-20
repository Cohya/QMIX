
from two_step_env import TwoStepEnv
import copy 
import numpy as np 
import matplotlib.pyplot as plt
from  Keras_network import ANNasClass, ANN
from Agent import Agent
from replay_memory import RandomMemory
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
    model = ANNasClass(input_shape = input_shape,
                output_shape = number_of_actions)
    
    target_model = ANNasClass(input_shape = input_shape,
                output_shape = number_of_actions)
    
    agent = Agent(model = model,
                  target_model = target_model,
                  i_d = i,
                  history = history)
    
    # fix target with main network weights
    agent.target_model.model.set_weights(model.model.get_weights())
    
    
    agents.append(agent)
                  
         
### Create the QMIX net 
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.RMSprop()
batch_size = 32
memory = RandomMemory(limit=500)
qmix = QMIX(
    agents=agents,
    replay_memory=memory,
    batch_size=batch_size,
    loss_fn=loss_fn,
    optimizer=optimizer)


episode_reward_history = []
loss_history = []
episode_reward_mean = 0
loss_mean = 0

max_episodes = 1000
s = env.reset()
init_state = tf.one_hot(s,observation_dims )

with tqdm.trange(max_episodes) as t:
    for episode in t:

        s = env.reset()
        init_state = tf.one_hot(s,observation_dims )
        for agent in agents:
            agent.reset(init_state)
        rewards = []
        for step in range(3): # <- max lingth of the game 
            actions = []
            for agent in agents:
                action = agent.get_action(eps  = 1.0)
                actions.append(action)
            state, reward, done = env.step(actions)
            state = tf.one_hot(state, observation_dims)
            rewards.append(reward)
            #!!!@Doto : change the logic of the game make the agent 
            # get the observation in the get action and delete the 
            # observ fucntion of it make the history out of the agent 
            trajectories = []
            for agent in agents:
                agent.observe(state) # The
                trajectory = copy.deepcopy(agent.trajectory)
                trajectories.append(trajectory)

            one_hot_actions = []
            for action in actions:
                action = tf.one_hot(action, depth=number_of_actions)
                one_hot_actions.append(action)
            qmix.save(state, trajectories, one_hot_actions, reward, done)

            if episode > batch_size:
                loss = qmix.train()
                loss_history.append(loss)

            if done:
                break

        episode_reward = np.sum(rewards)
        episode_reward_history.append(episode_reward)
        episode_reward_mean = 0.01 * episode_reward + 0.99 * episode_reward_mean
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
plt.savefig("result.png")