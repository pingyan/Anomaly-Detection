from __future__ import division, print_function, absolute_import
import sys,re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pprint 

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1

learning_rate_s,training_epochs_s,batch_size_s = sys.argv[3].split(",")
learning_rate = float(learning_rate_s)
training_epochs = int(training_epochs_s)
batch_size = int(batch_size_s)
#examples_to_test = 100

# load input as matrix
matfile = sys.argv[1]
testfile = sys.argv[2]
print("\n")
print("LOADING... INPUT MATRIX")
print("\n")
mat = np.load(matfile)
print("EXAMPLE INPUT")
pprint.pprint(mat[:5])
print("\n")
test = np.load(testfile)
#test = mat[1605]
pprint.pprint(test)
sampleCnt, n_input  = mat.shape
examples_to_test, _ = test.shape
print("Number of samples: ", sampleCnt)
print("Size of vocabulary: ", n_input)
print("\n")
#test = mat[:examples_to_test]
train = mat
trainCnt, _ = train.shape

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 128 # 2nd layer num features
#n_input = 784 # size of vocab

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    c_all = []
    total_batch = int(trainCnt/batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
	#np.random.shuffle(mat)
        for i in range(total_batch):
		np.random.shuffle(train)
		batch_xs = train[:batch_size]
		#pprint.pprint(batch_xs[1][:10])
		# Run optimization op (backprop) and cost op (to get loss value)
		_, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
		print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.5f}".format(c))
	c_all.append(c)
    outpath='-'.join( [learning_rate_s,training_epochs_s,batch_size_s])
    np.save("data/cost"+outpath, c_all)
    print("Optimization Finished!")
    # Applying encode and decode over test set
    encode_decode_oob = sess.run(
        y_pred, feed_dict={X: test})
    
    oob_err = []
    for i in range(examples_to_test):
	    rms = np.sqrt(((encode_decode_oob[i] - test[i]) ** 2).mean())
	    print ("Out-of-bag Test Seq's Squared Error", '%.9f' %rms)
	    oob_err.append(rms)
   
   # Compare original data points with their reconstructions
    train = mat[:examples_to_test]
    encode_decode_ib = sess.run(
		    y_pred, feed_dict={X: train})
    ib_err = []
    #for i in range(examples_to_test):
    ind_outlier = []
    for i in range(examples_to_test):
	    rms = np.sqrt(((encode_decode_ib[i] - train[i]) ** 2).mean())
	    #if rms > 100:
	    #print(','.join( ['{:.2f}'.format(x) for x in train[i]]))
	    print ("In-bag Test Seq's Squared Error", '%.9f' %rms)
	    ind_outlier.append(i)
	    ib_err.append(rms)
    print("oob_err: ", np.mean(oob_err), "ib_err", np.mean(ib_err))
    np.save("data/ooberr"+outpath,oob_err)
    np.save("data/iberr"+outpath,ib_err)
    #print(",".join(['{:04d}'.format(x) for x in ind_outlier]))
