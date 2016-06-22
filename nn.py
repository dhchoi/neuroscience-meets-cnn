import time
import pickle
import numpy as np
import tensorflow as tf
from data import load_data
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

num_steps = 6001
batch_size = 128
relu_units = 1024
SEED = 66478  # Set to None for random seed.


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def train(data, labels):
    train_size = data.shape[0]
    dim_size = data.shape[1]
    num_labels = labels.shape[1]

    num_train_end_index = (train_size / 9) * 8
    num_test_end_index = train_size

    train_dataset = data[0:num_train_end_index, :]
    train_labels = labels[0:num_train_end_index, :]
    test_dataset = data[num_train_end_index:num_test_end_index, :]
    test_labels = labels[num_train_end_index:num_test_end_index, :]

    print 'Training set', train_dataset.shape, train_labels.shape
    print 'Test set', test_dataset.shape, test_labels.shape

    graph = tf.Graph()
    with graph.as_default():
        # Input data. Using a placeholder that will be fed at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, dim_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        W1 = tf.Variable(tf.truncated_normal([dim_size, relu_units]))
        b1 = tf.Variable(tf.zeros([relu_units]))
        W2 = tf.Variable(tf.truncated_normal([relu_units, relu_units]))
        b2 = tf.Variable(tf.zeros([relu_units]))
        W3 = tf.Variable(tf.truncated_normal([relu_units, num_labels]))
        b3 = tf.Variable(tf.zeros([num_labels]))

        # We will replicate the model structure for the training subgraph, as well
        # as the evaluation subgraphs, while sharing the trainable parameters.
        def model(dataset, train=False):
            H1 = tf.nn.relu(tf.matmul(dataset, W1) + b1)
            H2 = tf.nn.relu(tf.matmul(H1, W2) + b2)
            # Add a 50% dropout during training only. Dropout also scales
            # activations such that no rescaling is needed at evaluation time.
            if train:
                H2 = tf.nn.dropout(H2, 0.5, seed=SEED)
            return tf.matmul(H2, W3) + b3

        # Training computation: logits + cross-entropy loss.
        logits = model(tf_train_dataset, True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(W1) + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(b3))
        loss += 5e-4 * regularizers  # Add the regularization term to the loss.

        # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            0.01,  # Base learning rate.
            batch * batch_size,  # Current index into the dataset.
            train_size,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

        # Predictions for the current training minibatch.
        train_prediction = tf.nn.softmax(logits)

        # Predictions for the test and validation, which we'll compute less often.
        test_prediction = tf.nn.softmax(model(tf_test_dataset))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print("Initialized")
        start_time = time.time()

        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)

            # Code for printing loss values for sanity check
            # print "tf.nn.l2_loss(W1).eval() * 5e-8", tf.nn.l2_loss(W1).eval() * 5e-8
            # print "tf.nn.l2_loss(b1).eval()", tf.nn.l2_loss(b1).eval()
            # print "tf.nn.l2_loss(W2).eval() * 5e-8", tf.nn.l2_loss(W2).eval() * 5e-8
            # print "tf.nn.l2_loss(b2).eval()", tf.nn.l2_loss(b2).eval()
            # print "tf.nn.l2_loss(W3).eval() * 5e-8", tf.nn.l2_loss(W3).eval() * 5e-8
            # print "tf.nn.l2_loss(b3).eval()", tf.nn.l2_loss(b3).eval()
            # regularizers = (tf.nn.l2_loss(W1) * 5e-8 + tf.nn.l2_loss(b1) + tf.nn.l2_loss(W2) * 5e-8 + tf.nn.l2_loss(b2) + tf.nn.l2_loss(W3) * 5e-8 + tf.nn.l2_loss(b3))
            # print "regularizers", regularizers.eval()
            # print "5e-4 * regularizers", 5e-4 * regularizers.eval()

            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
        print("Time taken:", int(time.time() - start_time))


if __name__ == "__main__":
    _, labels = load_data(True)
    # lda data (x_lda.p) was created using:
    #   data = LDA().fit(data.toarray(), labels.nonzero()[1]).transform(data.toarray())
    # pca data (x_pca.p) was created using:
    #   data = PCA().fit(data_X.toarray()).transform(data_X.toarray())
    data = pickle.load(open("x_lda.p", "rb")).astype(np.float32)

    n = 360  # trials per person
    num_subjects = 9  # num people
    for i in range(num_subjects):
        print "\n=== Using subject", i + 1, "as test set."
        data = np.concatenate((data[n:], data[:n]), axis=0)
        labels = np.concatenate((labels[n:], labels[:n]), axis=0)
        train(data, labels)

        
        
