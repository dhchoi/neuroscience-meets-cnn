import numpy as np
import tensorflow as tf
import pickle
import os, random
from data import load_data, convert_1d_to_3d, dim_x, dim_y, dim_z, dim_x_half
import pdb
import preprocess as prep
import time

num_labels = 60
num_steps = 100000

batch_size = 15
patch_size = 5
out_features1 = 32
out_features2 = 64
num_hidden = 1024
dropout = 0.5

def accuracy(predictions, labels):
    a = np.argmax(predictions,1)
    b = np.argmax(labels, 1)
    acc = 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]
    #tf.scalar_summary('accuracy/', acc)
    return acc


class ConvolutionalNetwork:
    def __init__(self, image_size_x, image_size_y, image_size_z):
        #self.image_size_x = image_size_x
        #self.image_size_y = image_size_y
        #self.image_size_z = image_size_z
        self.image_size_x = 25
        self.image_size_y = 61
        self.image_size_z = 23

    def reformat(self, dataset):
        """
        Reformat into a TensorFlow-friendly shape:
          - convolutions need the image data formatted as a cube (width by height by #channels)
          - labels as float 1-hot encodings.
        """
        #return dataset.reshape((-1, self.image_size_x, self.image_size_y, self.image_size_z, 1)).astype(np.float32)
        return dataset.reshape((-1, 21, 57, 20, 1)).astype(np.float32)

    def set_datasets(self, data, labels):
        num_total_data = len(data)
        #num_train_offset = num_total_data / 9 * 7
        num_train_offset = num_total_data / 6 * 5
        num_valid_offset = num_train_offset + (num_total_data / 6 )
        #num_valid_offset = num_train_offset
        num_test_offset = num_valid_offset

        self.train_dataset = (self.reformat(data[0:num_train_offset, :, :, :]))
        self.train_labels = labels[0:num_train_offset, :]
        #self.valid_dataset = (self.reformat(data[num_train_offset:num_valid_offset, :, :, :]))
        self.valid_dataset = (self.reformat(data[num_train_offset:num_valid_offset:50, :, :, :]))
        self.valid_labels = labels[num_train_offset:num_valid_offset:50, :]
        #self.test_dataset = (self.reformat(data[num_valid_offset:, :, :, :]))
        self.test_dataset = (self.reformat(data[num_valid_offset:-1:50, :, :, :]))
        self.test_labels = labels[num_valid_offset:-1:50, :]

        print 'Training set', self.train_dataset.shape, self.train_labels.shape
        print 'Validation set', self.valid_dataset.shape, self.valid_labels.shape
        print 'Test set', self.test_dataset.shape, self.test_labels.shape
        
    def train(self):
        """
        Two convolutional layers, followed by one fully connected layer, with stride of 2.
        """

        def weight_var(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

        def bias_var(shape):
            return tf.Variable(tf.constant(0.1, shape=shape))

        def conv3d(input, weights):
            return tf.nn.conv3d(input, weights, strides=[1,1,1,1,1], padding='SAME')

        def max_pool3d(input, k, s):
            return tf.nn.max_pool3d(input, ksize=[1,k,k,k,1], strides=[1,s,s,s,1], padding='SAME')

        def batch_norm(tensor):
            #m, v = tf.nn.moments(tensor, [0])
            m, v = tf.nn.moments(tensor, [0,1,2,3])
            return tf.nn.batch_normalization(tensor, m, v, None, None, 1e-12)
        
        graph = tf.Graph()

        with graph.as_default():
            # Input data.
            #tf_train_dataset = tf.placeholder(tf.float32,
            #                                  shape=(batch_size, self.image_size_x,
            #                                         self.image_size_y, self.image_size_z,1))
            train_shape = self.train_dataset.shape[1:]
            tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, train_shape[0], train_shape[1],
                                                                 train_shape[2], train_shape[3]))
            tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
            tf_valid_dataset = tf.constant(self.valid_dataset, dtype=tf.float32)
            tf_test_dataset = tf.constant(self.test_dataset, dtype=tf.float32)

            print tf_train_dataset.get_shape(), tf_valid_dataset.get_shape(), tf_test_dataset.get_shape()
            
            # Variables.
            # Layer 1 (conv)
            weights1 = weight_var([patch_size, patch_size, patch_size, 1, out_features1])
            biases1 = bias_var([out_features1])
            # Layer 2 (conv)
            weights2 = weight_var([patch_size, patch_size, patch_size, out_features1, out_features2])
            biases2 = bias_var([out_features2])
            # Layer 3 (fully connected)
            #weights3 = weight_var([self.image_size_x // 4 * self.image_size_y // 4
            #                       * self.image_size_z // 4 * out_features2, num_hidden])
            weights3 = weight_var([4032, num_hidden])
            biases3 = bias_var([num_hidden])
            # Layer 4 (output)
            weights4 = weight_var([num_hidden, num_labels])
            biases4 = bias_var([num_labels])

            # Model.
            def model(data, dropout=False):
                # do dropout on the input data itself i.e. blankout 
                #data = tf.nn.dropout(data, 0.5)

                # batch_normalization
                #data = batch_norm(data)
                
                # Layer 1 (conv): data = `patch_size` x `patch_size` x `image_size_z`
                print "conv1.shape (before conv)", data.get_shape()
                #pdb.set_trace()
                conv1 = conv3d(data, weights1)  # conv = `patch_size` x `patch_size` x `out_features1`
                conv1 = batch_norm(conv1)
                conv1 = tf.nn.relu(conv1 + biases1)
                print "conv1.shape (after conv)", conv1.get_shape()
                #conv1 = max_pool_kxk(conv1, 2)  # pool = `patch_size`/2 x `patch_size`/2 x `out_features1`
                conv1 = max_pool3d(conv1, 3,3)
                print "conv1.shape (after pool)", conv1.get_shape()
                if dropout:
                    conv1 = tf.nn.dropout(conv1, dropout)

                # Layer 2 (conv): data = `patch_size`/2 x `patch_size`/2 x `out_features1`
                print "conv2.shape (before conv)", conv1.get_shape()
                conv2 = conv3d(conv1, weights2)  # conv = `patch_size` x `patch_size` x `out_features2`
                conv2 = batch_norm(conv2)
                conv2 = tf.nn.relu(conv2 + biases2)
                print "conv2.shape (after conv)", conv2.get_shape()
                conv2 = max_pool3d(conv2, 3, 3)  # pool = `patch_size`/2/2 x `patch_size`/2/2 x `out_features2`
                print "conv2.shape (after pool)", conv2.get_shape()
                if dropout:
                    conv2 = tf.nn.dropout(conv2, dropout)

                # Layer 3 (fully connected): data = `patch_size`/2/2 x `patch_size`/2/2 x `out_features2`
                shape = conv2.get_shape().as_list()
                reshape = tf.reshape(conv2, [-1, shape[1] * shape[2] * shape[3] * shape[4]])  # reshape to fit dense layer input
                print "dense1.shape (input)", reshape.get_shape()
                print "dense1.shape (weights)", weights3.get_shape()
                dense1 = tf.nn.relu(tf.matmul(reshape, weights3) + biases3)
                #dense1 = batch_norm(dense1)
                if dropout:
                    dense1 = tf.nn.dropout(dense1, dropout)

                # Layer 4 (output)
                return tf.matmul(dense1, weights4) + biases4

            # Training computation.
            logits = model(tf_train_dataset)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

            # Optimizer.
            #self.optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(self.loss)
            self.optimizer = tf.train.AdamOptimizer(0.00005).minimize(self.loss)
            
            # Predictions for the training, validation, and test data.
            self.train_prediction = tf.nn.softmax(logits)
            correct_prediction = tf.equal(self.train_prediction, tf_train_labels)
            train_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('minibatch accuracy', train_acc)
            merged = tf.merge_all_summaries()
            
            self.valid_prediction = tf.nn.softmax(model(tf_valid_dataset, False))
            self.test_prediction = tf.nn.softmax(model(tf_test_dataset, False))

        with tf.Session(graph=graph) as session:
            #f_log = open('cropping.log' , 'wb')
            train_writer = tf.train.SummaryWriter('/home/ycho1/Desktop/', session.graph)
            tf.initialize_all_variables().run()
            print 'initialization done'
            for step in range(num_steps):
                offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
                batch_data = self.train_dataset[offset:(offset + batch_size), :, :, :]
                #batch_data = preprocess(batch_data)
                batch_labels = self.train_labels[offset:(offset + batch_size), :]
                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
                #_ , l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                summary, l, predictions = session.run([merged, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 50 == 0):
                    train_writer.add_summary(summary, step)
                    print('Minibatch loss at step %d: %f' % (step, l))
                    minibatch_acc = accuracy(predictions, batch_labels)
                    print('Minibatch accuracy: %.1f%%' % minibatch_acc)
                    val_acc = accuracy(self.valid_prediction.eval(), self.valid_labels)
                    print('Validation accuracy: %.1f%%' % val_acc)
                    #test_acc = accuracy(self.test_prediction.eval(), self.test_labels)
                    #print('Test accuracy: %.1f%%' % test_acc)
                    #f_log.write(('%d, %f, %.1f%% , %.1f%%, %.1f%% \n' % (step, l, minibatch_acc, val_acc, test_acc)))


if __name__ == "__main__":
    #data, labels = load_data(True)
    #data = pickle.load(open('./data/tot_x'))
    #labels = pickle.load(open('./data/tot_y'))

    data = pickle.load(open('data/ind_1_x', 'rb'))
    labels = pickle.load(open('data/ind_1_y', 'rb'))
    data = data.todense()
        
    data_3d = []  # data_3d.shape = (num_data, dim_x, dim_y, dim_z)
    data_3d = []  # data_3d.shape = (num_data, dim_x_half, dim_y, dim_z)
    for i in range(len(data)):
        d_3d = np.squeeze(np.asarray(data[i])).reshape((dim_x_half, dim_y, dim_z))
        #d_3d = np.squeeze(np.asarray(data[i])).reshape((21, 57, 20))
        data_3d.append(d_3d)
    data_3d = np.array(data_3d)

    t1 = time.time()
    data_3d, labels = prep.preprocess(data_3d, labels)
    t2 = time.time()
    print "%0.3f" % (t2-t1)
    
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    
    conv_net = ConvolutionalNetwork(dim_x_half, dim_y, dim_z)
    conv_net.set_datasets(data_3d, labels)
    conv_net.train()
