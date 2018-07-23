import numpy as np
import gym
import tensorflow as tf
env = gym.make('CartPole-v0')
env.reset()
n_inputs = 4  # == env.observation_space.shape[0]
n_hidden1 = 10 
n_hidden2 = 10
n_outputs = 1 # only outputs the probability of accelerating left
learning_rate = 0.01


X = tf.placeholder(tf.float32, shape=[None, n_inputs])
y = tf.placeholder(tf.float32, shape=[None, n_outputs])

hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer())
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, kernel_initializer=tf.glorot_normal_initializer())

logits = tf.layers.dense(hidden2, n_outputs)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cross_entropy)

saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     obs = env.reset()
#     
#     target_probas = np.array([([1.] if obs[2] < 0 else [0.])
#     
#     for step in range(1000):
#         print(step)
#         env.render()
#         action_val = action.eval(feed_dict={X:obs.reshape(1,n_inputs)})
#         obs, reward, done, info = env.step(action_val[0][0])
#         if done:
#             break
# env.close()n_environments = 10

n_environments = 50
envs = [gym.make("CartPole-v0") for _ in range(n_environments)]
observations = [env.reset() for env in envs]


#training parallel in 10 environments
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(200000):
        target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations]) # if angle<0 we want proba(left)=1., or else proba(left)=0.
        print(target_probas)
        
        action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(observations), y: target_probas})
        for env_index, env in enumerate(envs):
            obs, reward, done, info = env.step(action_val[env_index][0])
            observations[env_index] = obs if not done else env.reset()
    #saver.save(sess, "./my_policy_net_basic.ckpt")
    for env in envs:
        env.close()
# render policy net
    env = gym.make('CartPole-v0')
    observation = env.reset()
#with tf.Session() as sess:
#   saver.restore(sess, "./my_policy_net_basic.ckpt")
#  print('model loaded')
    for episode  in range(100):
        env.reset()
        for step in range(200):
            env.render()
            action_val = action.eval(feed_dict={X:observation.reshape(1, n_inputs)})
            observation, reward, done, info = env.step(action_val[0][0])
            
            if done:
                print("episode over: ", step)        
                break

