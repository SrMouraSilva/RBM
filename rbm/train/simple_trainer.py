from rbm.rbm import RBM

rbm = RBM(visible_size=28**2, hidden_size=10)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(mnist.train.labels)))
print("- Test-set:\t\t{}".format(len(mnist.test.labels)))
print("- Validation-set:\t{}".format(len(mnist.validation.labels)))

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

batch_size = 1
epochs = 10
num_tr_iter = int(mnist.train.num_examples / batch_size)

for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))

    for iteration in range(num_tr_iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x < 0.5  # Binarize
        print(batch_x, batch_y)

        # Run optimization op (backprop)
        feed_dict_batch = {x: batch_x, y: batch_y}
        session.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = session.run([loss, accuracy],
                                                feed_dict=feed_dict_batch)
            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))

    # Run validation after every epoch
    feed_dict_valid = {x: mnist.validation.images, y: mnist.validation.labels}
    loss_valid, acc_valid = session.run([loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')

session.close()