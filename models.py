import numpy as np
import matplotlib.pyplot as plt
import util


class Model:
    def __init__(self, layers, in_size, units, weight_init_range, bias_init_range=None, nonlins=None, loss="MSE",
                 lr=0.001, decay_rate=0, normalization=None, name="Basic NN Model"):
        """
        :param layers: number of layers in the network, not including input layer (must be at least one)
        :param in_size: number of inputs in the input layer
        :param units: np array of output units in each layer of the network, including input layer
        :param weight_init_range: support over which weights will be uniformly sampled from at initialization
        :param nonlins: (list of strings) list of non-linearities applied to each layer: 'linear', 'relu', 'sigmoid',
            'tanh', 'softmax'
        :param loss: loss function, options: "MSE", "CrossEntropy"
        :param lr: (float) learning rate for gradient descent
        :param normalization: normalization for the loss: options "NSE" (normalized square error), None
        :param name: (string) name of model
        """

        self.layers = layers
        self.units = units
        self.net = []
        self.activations = []
        self.deltas = []
        self.in_size = in_size
        self.activation_in = np.zeros(in_size)

        # list of non-linearities for each layer other than the input layer
        # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softmax'
        # defaults to linear layer
        self.nonlin_names = nonlins
        self.nonlins = []
        self.nonlins_backward = []
        self.loss_name = loss
        self.loss = util.lib_losses[loss]
        self.loss_backward = util.lib_losses_backward[loss]
        self.weight_init_range = weight_init_range
        self.lr = lr
        self.decay_rate = decay_rate
        self.weights = []
        self.biases = []
        self.bias_init_range = bias_init_range
        if bias_init_range is not None:
            self.have_bias = True
        else:
            self.have_bias = False
        self.train_loss = []
        self.normalization = normalization
        self.name = name

        # initialize weights
        self.weights.append(np.random.rand(self.units[0], self.in_size) *
                            (self.weight_init_range[1] - self.weight_init_range[0]) + self.weight_init_range[0])
        for idx in range(1, layers):
            self.weights.append(np.random.rand(self.units[idx], self.units[idx - 1]) *
                                (self.weight_init_range[1] - self.weight_init_range[0]) + self.weight_init_range[0])

        # initialize biases

        # set biases to zero, even if not in use
        for idx in range(self.layers):
            self.biases.append(np.zeros(self.units[idx]))

        # set biases if in use
        if self.have_bias:
            for idx in range(self.layers):
                self.biases[idx] = np.random.rand(self.units[idx]) * \
                                   (self.bias_init_range[1] - self.bias_init_range[0]) + self.bias_init_range[0]

        for idx in range(layers):
            self.net.append(np.zeros((self.units[idx], 1)))
            self.activations.append(np.zeros((self.units[idx], 1)))
            self.deltas.append(np.zeros((self.units[idx], 1)))
            if self.nonlin_names is None:
                self.nonlins.append(util.lib_nonlins['linear'])
                self.nonlins_backward.append(util.lib_nonlins_backward['linear'])
            else:
                self.nonlins.append(util.lib_nonlins[self.nonlin_names[idx]])
                self.nonlins_backward.append(util.lib_nonlins_backward[self.nonlin_names[idx]])

        # train steps taken
        self.steps = 0
        # train batches taken
        self.batch_steps = 0

    def forward(self, x):
        """
        :param self:
        :param x: input as np array
        :return: feedforward output as np array

        all layer activations stored in self.activations
        """
        # input layer
        self.activation_in = np.expand_dims(x, 1)
        # self.activation_in = x
        self.net[0] = np.matrix(self.weights[0]@self.activation_in + np.expand_dims(self.biases[0], 1))
        self.activations[0] = np.matrix(self.nonlins[0](self.net[0]))
        # feedforward through hidden and output layers
        for idx in range(1, self.layers):
            self.net[idx] = self.weights[idx]@self.activations[idx-1] + np.expand_dims(self.biases[idx], 1)
            self.activations[idx] = np.matrix(self.nonlins[idx](self.net[idx]))

        return self.activations[-1]

    def get_grad(self, y, y_hat):
        grad = [np.zeros(self.weights[layer].shape) for layer in range(self.layers)]

        dL_dout = self.loss_backward(y, y_hat)
        if self.nonlin_names[-1] == "softmax":
            self.deltas[-1] = self.nonlins_backward[-1](self.activations[-1], dL_dout)
        else:
            self.deltas[-1] = np.matrix(np.asarray(dL_dout) * np.asarray(self.nonlins_backward[-1](self.activations[-1])))
        grad[-1] = self.deltas[-1] @ self.activations[-2].T
        for idx in range(self.layers - 2, 0, -1):
            self.deltas[idx] = (self.weights[idx + 1].T @ self.deltas[idx + 1]) * self.nonlins_backward[idx](self.activations[idx])
            grad[idx] = self.deltas[idx] @ self.activations[idx - 1].T
        self.deltas[0] = np.matrix(np.asarray(self.weights[1].T @ self.deltas[1])
                                   * np.asarray(self.nonlins_backward[0](self.activations[0])))
        grad[0] = self.deltas[0] @ self.activation_in.T

        # bias derivatives
        if self.have_bias:
            for idx in range(self.layers):
                grad.append(self.deltas[idx])

        test1 = np.asarray(self.nonlins_backward[-1](self.activations[-1]))
        return grad

    def optimizer(self, grad):
        for idx in range(self.layers):
            self.weights[idx] += -self.lr * grad[idx] - self.decay_rate * self.weights[idx]

        # bias updates
        if self.have_bias:
            for idx in range(self.layers):
                self.biases[idx] += -self.lr * grad[idx + self.layers] - self.decay_rate * self.biases[idx]

    def train_step(self, x, y):
        """
        training algorithm
        :param x: np array of training examples, first dimension is batch dimension for batch mode
        :param y: np array of training labels, first dimension is batch dimension for batch mode
        :return:
        """
        batch_size = x.shape[0]
        self.train_loss.append(0)
        # include terms for biases
        grad = [np.zeros(self.weights[layer].shape) for layer in range(self.layers)]
        if self.have_bias:
            for idx in range(self.layers):
                grad.append(np.zeros(self.units[idx]))

        self.batch_steps += 1
        for idx in range(batch_size):
            self.steps += 1
            y_hat = self.forward(x[idx])
            # print(y_hat)
            loss = self.loss(y[idx], y_hat)
            sample_grad = self.get_grad(y[idx], y_hat)
            for layer in range(self.layers):
                grad[layer] += sample_grad[layer]
            self.train_loss[-1] += loss

        if self.normalization == "NSE":
            self.train_loss[-1] /= self.NSE_normalization(y)
        else:
            self.train_loss[-1] /= batch_size
        for layer in range(self.layers):
            grad[layer] /= batch_size
        if self.have_bias:
            for idx in range(self.layers):
                grad[idx + self.layers] /= batch_size
        self.optimizer(grad)

    def reset(self, reinit=True):
        self.train_loss = []
        for idx in range(self.layers):
            self.activations[idx] = np.zeros((self.units[idx], 1))
        self.steps = 0
        self.batch_steps = 0
        if reinit:
            self.reinitialize()

    def reinitialize(self):
        self.weights[0] = np.random.rand(self.units[0], self.in_size) * \
                            (self.weight_init_range[1] - self.weight_init_range[0]) + self.weight_init_range[0]
        for idx in range(1, self.layers):
            self.weights[idx] = np.random.rand(self.units[idx], self.units[idx - 1]) * \
                                (self.weight_init_range[1] - self.weight_init_range[0]) + self.weight_init_range[0]
        if self.have_bias:
            for idx in range(self.layers):
                self.biases[idx] = np.random.rand(self.units[idx]) * \
                                   (self.bias_init_range[1] - self.bias_init_range[0]) + self.bias_init_range[0]


    def NSE_normalization(self, labels):
        # normalized square error term
        label_sum = 0
        label_means = np.mean(labels, 1)
        for t in range(len(labels)):
            label_sum += np.dot(labels[t], labels[t])
        norm_factor = label_sum - np.dot(label_means, label_means) * labels.shape[0]
        return norm_factor

    def plot_train_loss(self):
        axis = plt.plot(self.train_loss)
        plt.yscale("log")
        plt.title("Training Loss")
        plt.xlabel("Samples")
        plt.ylabel("Loss")
        plt.show()

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name


class LillicrapModel(Model):
    def __init__(self, layers, in_size, units, weight_init_range, nonlins=None,
                 random_weights=False, randFB_init_range=None, lr=0.001, decay_rate=1e-06,
                 normalization="NSE"):
        """
        :param layers: number of layers in the network
        :param units: np array of output units in each layer of the network
        :param weight_init_range: support over which weights will be uniformly sampled from at initialization
        :param nonlins: list of non-linearities applied to each layer
        :param random_weights: whether the model will have random feedback weights (or will instead use basic GD)
        :param randFB_init_range: uniform sampling support for random feedback weights (if in use)
        """
        super(LillicrapModel, self).__init__(layers=layers, in_size=in_size, units=units, weight_init_range=weight_init_range,
                        nonlins=nonlins, lr=lr, decay_rate=decay_rate, normalization=normalization)

        self.random_weights = random_weights
        if self.random_weights is True:
            self.name = "Random Feedback Weights"
        else:
            self.name = "Backprop"

        self.B = []
        if random_weights is True:
            self.randFB_init_range = randFB_init_range
            for idx in range(layers - 1):
                self.B.append(np.random.rand(units[idx], units[idx + 1]) *
                                    (randFB_init_range[1] - randFB_init_range[0])
                              + weight_init_range[0])

    def loss(self, y, y_hat):
        e = np.expand_dims((y - y_hat).T, axis=1)
        return e, 0.5*np.dot(e.T, e)

    def get_grad(self, e, x):
        grad_W = -e@np.expand_dims(self.activations[1].T, 0)
        if self.random_weights is True:
            grad_W0 = -np.matrix(self.B[1])@e@np.matrix(x.T)
        else:
            grad_W0 = -self.weights[2].T@e@np.matrix(x.T)
        return grad_W0, grad_W

    def reinitialize(self):
        for idx in range(1, self.layers):
            self.weights[idx] = np.random.rand(self.units[idx], self.units[idx-1]) * \
                                (self.weight_init_range[1] - self.weight_init_range[0]) + self.weight_init_range[0]

        self.B = []
        if self.random_weights is True:
            for idx in range(1, self.layers):
                self.B.append(np.random.rand(self.units[idx-1], self.units[idx]) *
                                    (self.randFB_init_range[1] - self.randFB_init_range[0])
                              + self.weight_init_range[0])


# model with learning rule based on BCM Hebbian learning rule as described in
# "A synaptic basis for memory storage in the cerebral cortex" (Bear, 1996)
# Thanks to Dr. McClelland for his thoughts in developing this project
class BCMModel(Model):
    def __init__(self, layers, units, weight_init_range, nonlins=None,
                 random_weights=False, randFB_init_range=None,
                 lr=0.001, decay_rate=1e-06, normalization="NSE", BCM_decay_rate=0.9, BCM_sat_const=1):
        """
        :param layers: number of layers in the network
        :param units: np array of output units in each layer of the network
        :param weight_init_range: support over which weights will be uniformly sampled from at initialization
        :param nonlins: list of non-linearities applied to each layer
        :param random_weights: whether the model will have random feedback weights (or will instead have set feedback weights)
        :param randFB_init_range: uniform sampling support for random feedback weights (if in use)
        :param BCM_rate_decay: rate of decay in exponential moving average of BCM modification threshold
        """
        super(BCMModel, self).__init__(layers=layers, units=units, weight_init_range=weight_init_range,
                        nonlins=nonlins, lr=lr, decay_rate=decay_rate, normalization=normalization, name='BCM')

        self.BCM_decay_rate = BCM_decay_rate
        self.random_weights = random_weights
        self.randFB_init_range = randFB_init_range
        self.BCM_sat_const = BCM_sat_const

        # BCM neural modification thresholds
        self.q = []

        self.train_loss = []
        for idx in range(layers-1):
            self.weights.append(np.random.rand(units[idx+1], units[idx]) *
                                (weight_init_range[1] - weight_init_range[0]) + weight_init_range[0])
            self.q.append(np.zeros((self.units[idx], 1)))

        for idx in range(layers):
            self.activations.append(np.zeros((self.units[idx], 1)))

        self.B = []
        if random_weights is True:
            self.randFB_init_range = randFB_init_range
            for idx in range(layers - 1):
                self.B.append(np.random.rand(units[idx], units[idx + 1]) *
                              (randFB_init_range[1] - randFB_init_range[0])
                              + weight_init_range[0])

    def loss(self, y, y_hat):
        e = np.expand_dims((y - y_hat).T, axis=1)
        return e, 0.5*np.dot(e.T, e)

    # SIMPLE TEST WITH JUST BCM RULE FOR HIDDEN LAYER WEIGHTS WITH RANDOM FEEDBACK WEIGHTS,
    # ORIGINALLY WAS CONSIDERING CONSTANT FEEDBACK WEIGHTS, BUT RANDOM FEEDBACK WEIGHTS SHOULD
    # BE FINE AS WELL, POSSIBLY BETTER
    # ALSO WANT TO TEST PERFORMANCE WITH RANDOM FEEDBACK ON OUTPUT WEIGHTS, SIMPLE TO IMPLEMENT
    # WITH EXISTING RANDOM FEEDBACK WEIGHTS ALREADY BEING PRODUCED FOR ALL LAYERS
    def get_grad(self, e, x):
        grad_W = -e@np.expand_dims(self.activations[1].T, 0)
        if self.random_weights is True:
            grad_W0 = -np.matrix(self.B[1])@e@np.matrix(x.T)
            del_hidden = self.B[1] @ e
            # del_hidden[del_hidden<0] = 0
            post_syn_act = del_hidden
            # post_syn_act[post_syn_act<0] = 0
            # post_syn_act = del_hidden + self.activations[1][:, np.newaxis]
            # del_hidden = self.weights[1].T @ e

            # BASICALLY JUST AN APPLICATION OF MOMENTUM CONCEPT TO RANDOM FEEDBACK LEARNING RULE?
            # MORE OF A REVERSE APPLICATION, WE SUBTRACT EXPONENTIAL MEAN AS BCM THRESHOLD,
            # RATHER THAN USE IT IN UPDATE

            # BCM rule is applied to error signal input to neuron at distal apical dendritic compartment,
            # with the error signal in the GD framework corresponding to the gradient with respect to the activation
            self.q[0] = self.BCM_decay_rate * self.q[0] + (1 - self.BCM_decay_rate) * post_syn_act
            grad_W0 = -np.matrix(self.BCM_update_rule(post_syn_act, self.q[0])) @ np.matrix(x.T)
            # self.q[0] = self.BCM_decay_rate * self.q[0] + (1 - self.BCM_decay_rate) * post_syn_act
        else:
            grad_W0 = -self.weights[1].T@e@np.matrix(x.T)
        return grad_W0, grad_W

    def BCM_update_rule(self, post_syn_act, threshold):
        # num = post_syn_act * (post_syn_act - threshold)
        num = post_syn_act - threshold
        # return num
        # return num/(self.BCM_sat_const + num)
        return num/(np.maximum(self.BCM_sat_const + num, np.full(num.shape, self.BCM_sat_const)))

    def reinitialize(self):
        self.q = []
        for idx in range(1, self.layers):
            self.weights[idx] = np.random.rand(self.units[idx], self.units[idx-1]) * \
                                (self.weight_init_range[1] - self.weight_init_range[0]) + self.weight_init_range[0]
            self.q.append(np.zeros((self.units[idx], 1)))

        self.B = []
        if self.random_weights is True:
            for idx in range(1, self.layers):
                self.B.append(np.random.rand(self.units[idx-1], self.units[idx]) *
                                    (self.randFB_init_range[1] - self.randFB_init_range[0])
                              + self.weight_init_range[0])
