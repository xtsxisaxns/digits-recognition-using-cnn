from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt

# load csv format MNIST data
def load_digits(datasetPath):
    data = np.genfromtxt(datasetPath,delimiter = ",",dtype = "uint8")
    target = data[:,0]
    data = data[:,1:].reshape(data.shape[0],28,28)
    return (data,target)

print("Load training data")
(data,target) = load_digits("data/train.csv")
print('Training data shape:', data.shape, target.shape)
print("Data load complete")

# paramter setting
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

# image preprocessing,shape resize
def reformat(dataset, labels):
	dataset = dataset.reshape(
		(-1, image_size, image_size, num_channels)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

#data shuffle
indexShuffle = np.arange(data.shape[0])
np.random.shuffle(indexShuffle)
indexShuffle.reshape(data.shape[0],1)
data = data[indexShuffle,:,:]
target = target[indexShuffle]

train_dataset, train_labels = reformat(data[:30000,:,:], target[:30000])
valid_dataset, valid_labels = reformat(data[30001:35000,:,:], target[30001:35000])
test_dataset, test_labels = reformat(data[35001:,:,:], target[35001:])
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# define  accuracy
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])


# super paramter
batch_size = 16
patch_size = 5
depth1 = 32
depth2 = 64
num_hidden = 512


# dropout param
# keep_prob = 0.8

# l2_loss param
l2_beta = 5e-4


# begin define graph
graph = tf.Graph()

with graph.as_default():
	# Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    layer1_weights = tf.get_variable('w1',shape=[patch_size, patch_size, num_channels, depth1],initializer = tf.contrib.layers.xavier_initializer())
    layer1_biases = tf.Variable(tf.zeros(depth1))

    layer2_weights = tf.get_variable('w2',shape=[patch_size, patch_size, depth1, depth2],initializer = tf.contrib.layers.xavier_initializer())
    # layer2_biases = tf.Variable(tf.zeros(depth2))
    layer2_biases = tf.Variable(tf.constant(0.1,shape=[depth2]))

    layer3_weights = tf.get_variable('w3',shape=[image_size // 4 * image_size // 4 * depth2 , num_hidden],initializer = tf.contrib.layers.xavier_initializer())
    # layer3_biases = tf.Variable(tf.zeros(num_hidden))
    layer3_biases = tf.Variable(tf.constant(0.1,shape=[num_hidden]))

    layer4_weights = tf.get_variable('w4',shape=[num_hidden, num_labels],initializer=tf.contrib.layers.xavier_initializer())
    # layer4_biases = tf.Variable(tf.zeros(num_labels))
    layer4_biases = tf.Variable(tf.constant(0.1,shape=[num_labels]))


    # Model.
    def model(data,keep_prob = 1.0):
        # conv + max_pool
        conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        pool = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        hidden = tf.nn.relu(pool + layer1_biases)

        # conv + max_pool
        conv = tf.nn.conv2d(hidden, layer2_weights, [1,1,1, 1], padding='SAME')
        pool = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        hidden = tf.nn.relu(pool + layer2_biases)

        # conv + max_pool
        # conv = tf.nn.conv2d(hidden, new1_weights, [1,1,1,1], padding='SAME')
        # pool = tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        # hidden = tf.nn.relu(pool + new1_biases)

        # dropout
        hidden = tf.nn.dropout(hidden,keep_prob=keep_prob)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        # hidden = tf.nn.dropout(hidden, keep_prob=keep_prob)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # Training computation.
    logits = model(tf_train_dataset,1.0)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # l2_regularizer
    l2_regularizer = tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) + tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases)
    loss += l2_beta * l2_regularizer

    # learning rate dacay
    # global_step = tf.Variable(0)
    # # learning_rate = tf.train.exponential_decay(0.01,global_step,50000,0.95,staircase=True)
    # learning_rate = tf.train.exponential_decay(0.001, global_step, 50000, 0.95)
    # # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(loss,global_step=global_step)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)


    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model(tf_valid_dataset),valid_labels))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    # Save model for predict
    saver = tf.train.Saver()

# num_steps = 1001
#num_epochs * train_size) // BATCH_SIZE, E.g:10*200000//128
# num_steps = 15001
num_steps = 8000

# record loss and accuracy
train_loss_list = []
train_accu_list = []
valid_loss_list = []
valid_accu_list = []


with tf.Session(graph=graph) as session:
    #   tf.global_variables_initializer().run()
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in xrange(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print('--------step %d--------'% step)
            print('Minibatch loss :%f' % l)
            train_loss_list.append(l)
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            train_accu_list.append(accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            valid_loss_list.append(valid_loss.eval())
            valid_accu_list.append(accuracy(valid_prediction.eval(), valid_labels))
            print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

    #Save variables to disk
    savePath = saver.save(session,"model/cnn_model.ckpt")
    print("Model saved in file :%s"%savePath)

x = np.arange(0, len(train_accu_list) * 100, 100)
plt.figure()
plt.plot(x, train_accu_list, 'g', label='train accuracy')
plt.plot(x, valid_accu_list, 'r', label='validation accuracy')
plt.legend()

plt.figure()
plt.plot(x,train_loss_list,'g',label='train loss')
plt.plot(x,valid_loss_list,'r',label='validation loss')
plt.legend()
plt.show()
