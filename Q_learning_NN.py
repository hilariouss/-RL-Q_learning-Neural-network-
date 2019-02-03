# input : 1 x 16
# output : for each input, 4 output is valid.

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

##########################################
# 1. Neural network construction
##########################################

tf.reset_default_graph()

# 1-1) input/weight of neural network
input = tf.placeholder(shape=[1, 16], dtype=tf.float32)
weight = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
# tf.random_uniform : shape/minval/maxval --> generate random value under 'uniform distribution', between minval and maxval.

Qout = tf.matmul(input, weight) # output of neural network (1 x 4) ==> PREDICTED Q

predict = tf.argmax(Qout, 1) # Choose the most promising action (row)

# 1-2) Get loss with 'target' Q value and 'predicted' Q value.
nextQ = tf.placeholder(shape=[1, 4], dtype=tf.float32) # TARGET Q
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)


##########################################
# 2. Teach the constructed neural network
##########################################
init = tf.global_variables_initializer()

discount_factor = 0.99 #
e = 0.1 # stands for e-greedy.

num_episodes = 2000
# Create list (total reward, step)
stepList = []
RewardList = []
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset the environment and get first observation.
        s = env.reset()
        Total_reward = 0
        done = False
        step = 0
        # Q network
        while step < 99:
            step += 1
            # Randomly choose action
            action, allQ = sess.run([predict, Qout], \
                                    feed_dict={input:np.identity(16)[s:s+1]})

            if np.random.rand(1) < e: # greedy action
                action[0] = env.action_space.sample()

            # Get new state and reward
            s_, reward, done, _ = env.step(action[0])
            # Input the new state and get Q_
            Q_ = sess.run(Qout, feed_dict={input:np.identity(16)[s_:s_+1]})

            # Get max Q_ and set the action corresponding to the target Q_ value
            maxQ_ = np.max(Q_)
            targetQ = allQ

            targetQ[0,action[0]] = reward + discount_factor*maxQ_
            #update the network using the target value
            _, w_ = sess.run([updateModel, weight], \
                             feed_dict={input:np.identity(16)[s:s+1], nextQ:targetQ})
            Total_reward += reward
            s = s_
            if done == True:
                e = 1./((i/50)+10)
                break
        stepList.append(step)
        RewardList.append(Total_reward)

##########################################
# 3. Check out the performance with 'pyplot'.
##########################################
print("Percent of successful episodes: " + str(sum(RewardList)/num_episodes))
plt.plot(RewardList)