from models import *

import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
num_steps = 10000
batch_size = 256

display_step = 1000
examples_to_show = 10


model = AE(None, )
# model = SAE(obs_dim=784, h_size=[64], lr=1e-2)
# model = DAE(obs_dim=784, h_size=64, lr=1e-2, drop_prob=.2, noise_type="zeros")
# model = SAE(obs_dim=784, h_size=(128,64), lr = 1e-2)
params = {
    "obs_dim": 784,
    "h_size": (128, 64),
    "lr": 1e-2
}
# use this to test Stacked
# trainer = Trainer(enc_type="SAE", network_params=params)
#
# loss = trainer.train(dataset=mnist.train, max_steps=num_steps, batch_size=32)
# trainer.save()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(1, num_steps + 1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([model.train_op, model.loss], feed_dict={model.x: batch_x}) # add this for deonising, model._is_training: True})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(model.x_hat, feed_dict={model.x: batch_x}) # model._is_training: False})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()
