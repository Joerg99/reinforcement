import gym
import tensorflow as tf
env = gym.make('CartPole-v0')

n_inputs = 4  # == env.observation_space.shape[0]
n_hidden = 4  # it's a simple task, we don't need more than this
n_outputs = 1 # only outputs the probability of accelerating left
initializer = tf.contrib.layers.variance_scaling_initializer()

# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
                         kernel_initializer=initializer)
outputs = tf.layers.dense(hidden, n_outputs, activation=tf.nn.sigmoid,
                          kernel_initializer=initializer)

# 3. Select a random action based on the estimated probabilities
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    obs = env.reset()
    for step in range(1000):
        action_val = action.eval(feed_dict={X:obs.reshape(1,n_inputs)})
        obs, reward, done, info = env.step(action_val[0][0])


for i_episode in range(4):
    observation = env.reset()
    for t in range(500):
        env.render()
        #print(observation)
        position, velocity, angle, angular_velocity  = observation
#         if angle < 0:
#             action = 0
#         else:
#             action = 1
             
        observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break

