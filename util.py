import numpy as np
import matplotlib.pyplot as plt


def run(models, data_loader, n_runs=1, n_epochs=1, n_samples=100, batch_size=1, test=False):
    """
    :param models: models for training comparison
    :param data_loader: object for loading and sampling data from test data set
    :param runset: number of runs to average over
    :param n_epochs: number of training epochs
    :param batchsize: number of samples per batch
    :return:
        mean_loss: list of mean (over runs) of losses over training for each model
    """

    mean_loss = [0 for _ in range(len(models))]
    n_correct_max = [0]

    # record number of failed runs with exploding gradients
    run_failures = np.zeros(len(models))
    n_batches = n_samples // batch_size

    for run_i in range(n_runs):
        print("Run {}".format(run_i))
        # samples, labels = data_loader.sample(n_samples)
        for i in range(len(models)):
            models[i].reset()
            print("Model {}: {}".format(i, models[i].get_name()))
            for epoch in range(n_epochs):
                samples, labels = data_loader.sample(n_samples)
                for batch_i in range(0, n_samples, batch_size):
                    x = samples[batch_i:(batch_i+batch_size)]
                    y = labels[batch_i:(batch_i+batch_size)]
                    models[i].train_step(x, y)
                print("\tEpoch {}, Loss: {}".format(epoch, np.mean(models[i].train_loss[-batch_size:])))
                if test:
                    n_correct = 0
                    # test = data_loader.labels_test[0]
                    for idx in range(data_loader.examples_test.shape[0]):
                        y_hat = models[i].forward(data_loader.examples_test[idx])
                        if np.argmax(y_hat) == np.argmax(data_loader.labels_test[idx]):
                            n_correct += 1
                            n_correct_max[0] = max(n_correct_max[0], n_correct)
                            if n_correct < 0.6*n_correct_max[0]:
                               break
                    print('\t\tTest results: {}/{}'.format(n_correct, data_loader.examples_test.shape[0]))
            # check for divergent training run
            if not np.any(np.isnan(np.array(models[i].train_loss))):
                mean_loss[i] += np.array(models[i].train_loss)
            else:
                run_failures[i] += 1

    print("Divergent runs in training:")
    print(run_failures)

    mean_loss = [mean_loss[i]/n_runs for i in range(len(models))]

    return mean_loss


class DataLoader:
    """
    DataLoader loads and samples data from given data set
    """

    def __init__(self, examples=None, labels=None, examples_test=None, labels_test=None, only_samples=False, sampler=None):
        """

        :param examples: np array of data examples
        :param labels: np array of data labels
        :param examples_test: np array of test examples
        :param labels_test: np array of test labels
        :param only_samples: Boolean indicating whether the DataLoader generates synthetic data with a function "sampler"
        :param sampler: synthetic data-generating function
        """
        self.only_samples = only_samples
        if not self.only_samples:
            self.data_set = (examples, labels)
        else:
            self.sampler = sampler

        self.examples_test = examples_test
        self.labels_test = labels_test

    def sample(self, n_samples):
        if not self.only_samples:
            # PROBABLY WANT TO REPLACE THIS WITH A REGULAR SHUFFLE THAT OCCURS ONCE THE DATA HAS BEEN COMPLETELY RUN THROUGH AS WELL AS WITH AN EXTERNAL FUNCTION CALL,
            # ALTHOUGH COULD BE SLOW...
            # return self.data_set[0], self.data_set[1]
            sample_idx = np.random.choice(np.array([i for i in range(len(self.data_set[0]))]), size=n_samples, replace=False)
            return self.data_set[0][sample_idx], self.data_set[1][sample_idx]
        return self.sampler(n_samples)

    def load_data(self):
        return self.data_set


def plot_loss(losses, models):
    for i in range(len(losses)):
        plt.subplot(len(losses), 1, i + 1)
        plt.plot(losses[i])
        plt.yscale("log")
    plt.subplot(len(losses), 1, 1)
    plt.title("Training Loss (NSE)")
    plt.ylabel("Loss")

    plt.figure(2)
    plt.title("Training Loss (NSE)")
    plt.xlabel("No. Samples")
    plt.ylabel("Loss")
    # losses[i] = losses[i].reshape()
    for i in range(len(models)):
        plt.plot(losses[i], label=models[i].get_name())
        plt.yscale("log")
    plt.legend()

    plt.show()


def relu(x):
    x[x < 0] = 0
    return x


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


def linear(x):
    return x


def softmax(x):
    x = np.copy(x)
    x -= np.max(x)
    x = np.exp(x)
    return x/np.sum(x)


def tanh(x):
    exp_plus = np.exp(x)
    exp_minus = np.exp(-x)
    return (exp_plus - exp_minus)/(exp_plus + exp_minus)


lib_nonlins = {
    'relu': relu,
    'sigmoid': sigmoid,
    'softmax': softmax,
    'tanh': tanh,
    'linear': linear
}


def relu_backward(act):
    grad = np.copy(act)
    grad[grad < 0] = 0
    grad[grad >= 0] = 1
    return grad


def sigmoid_backward(act):
    return np.matrix(np.asarray(act) * np.asarray(1-act))


def linear_backward(act):
    return np.ones(act.shape)


def softmax_backward(act, grad_loss):
    jac = -act @ act.T
    for i in range(act.shape[0]):
        jac[i, i] += act[i]
    delta = jac.T @ grad_loss
    return delta


def tanh_backward(act):
    return 1 - act*act


lib_nonlins_backward = {
    'relu': relu_backward,
    'sigmoid': sigmoid_backward,
    'softmax': softmax_backward,
    'tanh': tanh_backward,
    'linear': linear_backward
}


def MSE(target, output):
    diff = np.expand_dims(target, 1) - output
    loss = 0.5*np.dot(diff.T, diff)[0][0].item()
    return loss
    # return 0.5*np.dot((np.expand_dims(target, 1) - output).T, np.expand_dims(target, 1) - output)[0][0].item()


def MSE_backward(target, output):
    return -(np.expand_dims(target, 1) - output)


# cross entropy for one-hot targets and for probabilistic outputs
def CrossEntropy(target, output):
    return -np.log(output[np.argmax(target)])[0][0].item()


def CrossEntropy_backward(target, output):
    loss_backward = np.zeros(output.shape)
    loss_backward[np.argmax(target)] = -1/output[np.argmax(target)]
    return loss_backward


lib_losses = {
    "MSE": MSE,
    "CrossEntropy": CrossEntropy
}


lib_losses_backward = {
    "MSE": MSE_backward,
    "CrossEntropy": CrossEntropy_backward
}


# originally developed by aldro61: https://gist.github.com/aldro61/40233cb59a3acf725dde6abb617141d4
def load_mnist():
    """
    Load the MNIST dataset into numpy arrays
    Author: Alexandre Drouin
    License: BSD
    """
    import numpy as np

    from tensorflow.examples.tutorials.mnist import input_data


    mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

    X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
    y_train = mnist.train.labels

    X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
    y_test = mnist.test.labels

    del mnist
    return X_train, y_train, X_test, y_test
