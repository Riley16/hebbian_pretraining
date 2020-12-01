from models import LillicrapModel
import nn_fun.tests
import util
import numpy as np
import time
import matplotlib.pyplot as plt

seed = 2
np.random.seed(seed)


if __name__ == '__main__':
    # run_args = tests.linear_target_Lillicrap()
    models = []
    losses = []

    run_args = nn_fun.tests.linear_target_Lillicrap(seed=seed)
    models += run_args[0]
    losses += util.run(*run_args)
    t_start = time.time()

    # run_args = tests.MNIST_basic(seed=seed)
    # # run_args = tests.linear_target_basic_GD_model_class_test(seed=seed)
    # models += run_args[0]
    # # losses += util.run(*run_args)
    # losses += util.run(*run_args, test=True)

    t_end = time.time()
    print('Time elapsed during run(s) (training and testing): {}'.format(t_end - t_start))

    # BCM_decay_rates = [0.9]
    # # BCM_decay_rates = [0.1, 0.5, 0.9, 0.99]
    # for i in range(len(BCM_decay_rates)):
    #     # print(hyperparameter value...)
    #     run_args = tests.linear_target_BCM(seed=seed, BCM_decay_rate=BCM_decay_rates[i], BCM_sat_const=1)
    #     run_args[0][0].set_name(run_args[0][0].get_name() + ", rate = {}".format(BCM_decay_rates[i]))
    #     models += run_args[0]
    #     losses += util.run(*run_args)

    # BCM_sat_consts = [0.8, 1, 5, 10, 100]
    # for i in range(len(BCM_sat_consts)):
    #     run_args = tests.linear_target_BCM(seed=seed, BCM_decay_rate=0.9, BCM_sat_const=BCM_sat_consts[i])
    #     run_args[0][0].set_name(run_args[0][0].get_name() + ", k = {}".format(BCM_sat_consts[i]))
    #     models += run_args[0]
    #     losses += util.run(*run_args)

    util.plot_loss(losses, models)

"""
models = []

# basic GD model
# not entirely clear what the original hyperparameters used were in Lillicrap paper for the linear target learning task
in_size = 30
out_size = 10
units = (in_size, 20, out_size)
layers = len(units)

# learning rate of 0.005 (true rate unspecified) seems to give performance similar to Lillicrap et al
# "Random synaptic feedback weights support error backpropagation for deep learning" Fig. 2 (a).
lilli_GD = LillicrapModel(layers=layers, units=units, weight_init_range=(-0.01, 0.01), lr=0.005, decay_rate=0)
models.append(lilli_GD)

lilli_randFB = LillicrapModel(layers=layers, units=units, weight_init_range=(-0.01, 0.01), lr=0.005, decay_rate=0,
                              random_weights=True, randFB_init_range=(-0.5, 0.5))
models.append(lilli_randFB)

n_samples = 2000
batch_size = 1
n_batches = n_samples//batch_size

# record number of failed runs with exploding gradients
run_failures = np.zeros(len(models))

epochs = 1
runset = 20
mean_loss = [0 for _ in range(len(models))]
for run in range(runset):
    print("Run {}".format(run))
    # linear target function
    LinFunc = LillicrapModel(layers=2, units=(in_size, out_size), weight_init_range=(-1, 1))
    samples = np.random.multivariate_normal(np.zeros(in_size), np.eye(in_size), n_samples)
    labels = np.array([LinFunc.forward(sample) for sample in samples])

    for i in range(len(models)):
        models[i].reset()
        print("Model {}".format(i))
        for epoch in range(epochs):
            print("\tepoch {}".format(epoch))
            for batch_i in range(0, n_samples, batch_size):
                x = samples[batch_i:(batch_i+batch_size)]
                y = labels[batch_i:(batch_i+batch_size)]
                models[i].train_step(x, y)
        # check for divergence
        if not np.any(np.isnan(np.array(models[i].train_loss))):
            mean_loss[i] += np.array(models[i].train_loss)
        else:
            run_failures[i] += 1

print("Divergent runs in training:")
print(run_failures)

mean_loss = [mean_loss[i]/runset for i in range(len(models))]

for i in range(len(models)):
    plt.subplot(len(models), 1, i+1)
    plt.plot(mean_loss[i])
    plt.yscale("log")
# plt.plot(mean_loss[0])
plt.subplot(len(models), 1, 1)
plt.title("Training Loss (NSE)")
plt.ylabel("Loss")

plt.figure(2)
plt.title("Training Loss (NSE)")
plt.xlabel("No. Samples")
plt.ylabel("Loss")
for i in range(len(models)):
    plt.plot(mean_loss[i], label=models[i].get_name())
    plt.yscale("log")
plt.legend()

plt.show()
"""

