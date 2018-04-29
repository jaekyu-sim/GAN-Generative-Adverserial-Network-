
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

"""def make_batch(input_data, noise_data, batch_size):
    index = np.arange(0, len(input_data))
    np.random.shuffle(index)
    index = index[:batch_size]
    shuffled_input_data = [input_data[i] for i in index]
    shuffled_noise_data = [noise_data[i] for i in index]
    
    return np.asarray(shuffled_input_data), np.asarray(shuffled_noise_data)"""
def make_batch(noise_data, batch_size):
    index = np.arange(0, len(noise_data))
    np.random.shuffle(index)
    index = index[:batch_size]
    shuffled_noise_data = [noise_data[i] for i in index]
    
    return np.asarray(shuffled_noise_data)
def make_noise_vector(batch_size, noise_size):
    noise_vector = np.random.normal(size=(batch_size, noise_size))
    return noise_vector


# In[3]:

class GAN(object):
    def __init__(self, sess, batch_size):
        self.sess = sess
        self.batch_size = batch_size
        self.Noise_Input_Data_Size = 128#32
        self.Hidden_Layer1_Size_G = 256
        self.Hidden_Layer1_Size_D = 256
        self.Converted_Image_Size = 784
        self.parameter()
        self.model()
        
    def parameter(self):
        #input parameter
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.Converted_Image_Size])
        self.Z = tf.placeholder(dtype=tf.float32, shape=[None, self.Noise_Input_Data_Size])
                
        #generator parameter
        self.W1_G = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[self.Noise_Input_Data_Size, self.Hidden_Layer1_Size_G], stddev=0.01))
        self.b1_G = tf.Variable(tf.zeros(dtype=tf.float32, shape=[self.Hidden_Layer1_Size_G]))
        self.W2_G = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[self.Hidden_Layer1_Size_G, self.Converted_Image_Size], stddev=0.01))
        
        #discriminator parameter
        self.W1_D = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[self.Converted_Image_Size, self.Hidden_Layer1_Size_D], stddev=0.01))
        self.b1_D = tf.Variable(tf.zeros(dtype=tf.float32, shape=[self.Hidden_Layer1_Size_D]))
        self.W2_D = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[self.Hidden_Layer1_Size_D, 1], stddev=0.01))
        
    def generator(self, noise):
        self.L1_G = tf.add(tf.matmul(noise, self.W1_G), self.b1_G)
        self.Y1_G = tf.nn.relu(self.L1_G)
        self.model_G = tf.nn.sigmoid(tf.matmul(self.Y1_G, self.W2_G))
                
        return self.model_G
    
    def discriminator(self, input_data):
        self.L1_D = tf.add(tf.matmul(input_data, self.W1_D), self.b1_D)
        self.Y1_D = tf.nn.relu(self.L1_D)
        self.model_D = tf.nn.sigmoid(tf.matmul(self.Y1_D, self.W2_D))
        
        return self.model_D
    
    def model(self):
        noise_data = make_noise_vector(self.batch_size, noise_size = self.Noise_Input_Data_Size)
        self.G = self.generator(self.Z)
        self.D_fake = self.discriminator(self.G)
        self.D_real = self.discriminator(self.X)
        
        D_var_list = [self.W1_D, self.b1_D, self.W2_D]
        G_var_list = [self.W1_G, self.b1_G, self.W2_G]
        
        self.cost_D = tf.reduce_mean(tf.log(self.D_real) + tf.log(1-self.D_fake))#maximize
        self.Optimize_D = tf.train.AdamOptimizer(learning_rate=0.002).minimize(-self.cost_D, var_list = D_var_list)
        #self.Optimize_D = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-self.cost_D)
        #self.cost_G = tf.reduce_mean(tf.log(self.D_real))#maximize
        self.cost_G = tf.reduce_mean(tf.log(1-self.D_fake))#minimize
        self.Optimize_G = tf.train.AdamOptimizer(learning_rate=0.002).minimize(self.cost_G, var_list = G_var_list)
        #self.Optimize_G = tf.train.AdamOptimizer(learning_rate=0.001).minimize(-self.cost_G)
        
    def training(self):
        data_size = 55000
        
        total_batch = int(data_size / self.batch_size)
        #SAVE_PATH = "C:/Users/JAEKYU/Documents/Jupyter Notebook/Super_Resolution_/Weight/Weight.ckpt"

        print("Session start")
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(40):
            for i in range(total_batch):
                #batch 만들기
                batch_X = mnist.train.next_batch(batch_size=self.batch_size)[0]#32,784
                batch_Z = make_batch(make_noise_vector(self.batch_size, self.Noise_Input_Data_Size), self.batch_size)#32,10
                #print("batch 생성 end")
                #session run
                #print("sesstion run start")
                Opt_G, cost_G = self.sess.run([self.Optimize_G, self.cost_G], feed_dict={self.Z : batch_Z})
                Opt_D, cost_D = self.sess.run([self.Optimize_D, self.cost_D], feed_dict={self.X : batch_X, self.Z : batch_Z})
                #print("sesstion run sucess")
            print("epoch : ", epoch, ", gen_cost : ", cost_G, ", dis_cost : ", cost_D)
            
            

            if(epoch%10 == 0):
                noise_data = make_batch(make_noise_vector(self.batch_size, self.Noise_Input_Data_Size), self.batch_size)
                samples = self.sess.run(self.G, feed_dict={self.Z : noise_data})
                fig, ax = plt.subplots(1, self.batch_size, figsize=(self.batch_size, 1))
                for j in range(self.batch_size):
                    ax[j].set_axis_off()
                    ax[j].imshow(np.reshape(samples[j], (28, 28)))
                fig.show()
                plt.draw()
                plt.show()
                

            


# In[4]:




# In[ ]:



