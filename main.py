from models import LillicrapModel
import tests
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

    # run_args = tests.linear_target_Lillicrap(seed=seed)
    # models += run_args[0]
    # losses += util.run(*run_args)
    t_start = time.time()

    run_args = tests.MNIST_basic(seed=seed)
    # run_args = tests.linear_target_basic_GD_model_class_test(seed=seed)
    models += run_args[0]
    # losses += util.run(*run_args)
    losses += util.run(*run_args, test=True)

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
TODO
write simple correlation error tests (with one neuron conforming to target, also XOR learning task) demonstrating basic
capabilities of the CorrNet

write linear -> sigmoid class with access to linear weights, forward operation, along with distal apical dendritic 
compartment with learning step update
write composite class with FF layers, check into input linking

implement feedback neurons which build up connection strengths, then inhibit internal response of FF network
write FB network to connect to each neuron in FF net

consider writing BCM modification thresholding rule


SHOULD MAKE EXPO MEAN NOT INCLUDE CURRENT TIME SAMPLE
BCM learning rule:
    net input term in post-synaptic activity
        try different weightings of net/error terms, try pure Hebbian learning for building up initial unsupervised 
        structure
            try annealing between Hebbian learning and GD by interpolation schedule between net input term and error 
            term in post-synaptic activity
            try fully switching between the two, similar to periods of learning with and without error...
            measure degree of input-output correlation, degree of stimulus selectivity, try switching on a schedule 
            based on signal selectivity or some quantity of the weights, like weight variance in a single neuron
            try on more complex models, may need to alter schedule (try alternating between the two) for non-linear 
            neurons
            test learning with and without Hebbian learning session, with Hebbian learning sessions happening midway 
            through regular GD training, even in combination with them, maybe allow for post-synaptic activity due to 
            respective signal to build up respectively
        try a learning rule based on recognizing neurons with high error correlation (take 10 neurons with highest error 
        correlation, or N neurons that acocunt for some percent of the error variance, see article on analyzing NNs with
        Gini coefficient), other metrics of sparsity of responsiveness, selectivity, don't want the neurons to become 
        too selective, look into inhibitive effects which could prevent this 
        and focusing on optimizing critical
        "key" neurons, neurons that act as hinges, maybe use noise to get over local "hills", 
            try neural noise as a percentage of neural post-synaptic activity, as well as a percentage of non-linearity 
            range
        
        
        what effects could adding the net input term have on learning? could it improve learning in some way with proper 
        weighting? 
            net average alone as post-synaptic response tends to cause input separation
            error signal alone gives random FB GD with BP
            
        
        
        could net input be relevant for biologically realistic learning scenarios in which errors signals may not always 
        be strong? 
            compare with Guerguiev et al in which FF computation first without error signal and then with error signal
            is allowed to propagate (might be a misinterpretation. in general FF propagation occurs, 
            followed by a delay, and then the error signal arrives, giving the necessary difference). If no error signal arrives, no difference,
            no weight changes
            say error signals are temporally sparse, not always present, say net average
    zero tapering
        multiplying by net input? multiplying by net input + error term?

gradient smearing: 
    dropped from project for now
    
    topologically, the random FB method of Lillicrap seems artificial with orderly layers of FB weights going back with
    symmetric topology to FF network. Symmetric weights are removed, but symmetric topology is not. How realistic is the
    symmetric topology? How can it be broken up?
    GD vs. BP? how can BP be partially implemented for GD in a biologically plausible way? 
        random FB weights answer question of symmetric weights
        how can problem of symmetric topology be solved? does it need to be addressed?



    sparsifying the FB matrices
        random sparisifying - may require new initialization range
        selective sparsifying by creating FB channels (chains of FB neurons that connect back along through that chain, 
        rather than through the entire layer) that connect with each other some
        need to ensure error percolates back, not too many connections to non-existent neurons.
            
    trying random FB matrices connecting directly from the error signal to all layers
        try random FB matrix to output layer
    

restricting net activation to positive values with all positive weights in target function as well as all positively 
weighted targets and input values
playing with net input


BCM learning rule model
    CONSIDER SYSTEMATIC METHOD FOR EXPLORING HYPERPARAMETER AND LEARNING RULE SPACE
    WHAT DO I CARE ABOUT FOR THIS PROJECT? BEATING BACKPROP? OR MAKING A BIOLOGICALLY PLAUSIBLE
    ALGORITHM? AT THIS POINT I'M NOT REALLY USING THE BCM LEARNING RULE, NEED TO EXPLORE 
    PERFORMANCE OF SEVERAL BCM VARIANTS
    
        EXPONENTIAL MEAN VS. REGULAR MEAN (BUT REGULAR MEAN IS NOT PLAUSIBLE), 

        FULL BCM UPDATE RULE VS. SIMPLIFICATIONS WITHOUT MULTIPLYING BY NET
        NEED TO ALSO CONSIDER ADDING NET TERM INTO POST-SYNAPTIC ACTIVITY
        NEED TO CONSIDER FULLER PICTURE WITH NEGATIVE WEIGHTS FOR LINEAR TARGET FUNCTIONS
        TRY BCM RULES WITH ALL LAYERS RATHER THAN JUST HIDDEN LAYER, TRY MORE LAYERS ETC.
        FINALLY TRY FULL ON REWARD SIGNAL RATHER THAN RANDOM FEEDBACK WEIGHT APPROACH (I.E. 
        SET RANDOM FEEDBACK WEIGHTS TO 1/OUT_SIZE)
        could choices of hyperparameters basically just mean cause gradient updates to converge to GD results? 
        i.e. the learning rule isn't doing anything other than getting out of the way
        the current BCM learning rule subtracts an exponential moving average of the error signals from the current
        error
        could it be that as decay rate increases, the expoential moving average averages out to zero, 
        resulting in just the ordinary gradient from having an effect? but that would result in the effective learning 
        rate being reduced by a factor of (1 - BCM_decay_rate)... 
        also the threshold basically subtracts each error term many times, check into the total effective values 
        subtracted, seems to be undoing the work of each update, at least partially, with lower decay giving 
        faster learning... 
            could be that subtraction of previous error term in a sense removes commitment to old weights, which are 
            
        COULD ALSO BE THE CASE THAT REQUIREMENT THAT POSTSYNAPTIC ACTIVITY IN BCM RULE BE POSITIVE
        COULD ALSO MEAN THAT USING SATURATION FACTOR IS IRRELEVANT BECAUSE IT'S SET TO ONE MOST OF THE TIME...
        CHECK SIGNS OF DERIVATIVES
        
        examine what happens with BCM when total postsynaptic activity is negative, 
        see if enforcing non-negative postsynaptic activity causes poor effects
            what is BCM learning rule for negative post-synaptic activity? should cause no weight change
        
        could tune BCM rule curve (hard, threshold is dynamic, not controlled, max function in denominator always resets
        at zero, ruining attempts at scaleless tuning)
            explicitly eliminate singularities with piecewise functions?
        
        try enforcing minimum absolute activation in BCM denominator
        try enforcing non-negative postsynaptic activity
        
        ANALYZE THE ALGORITHM, VERIFY THAT IT IS NON-TRIVIAL, THAT IT IS NOT JUST FOLLOWING
        RANDOM FEEDBACK WEIGHTS
        
        ALSO NEED MORE IN-DEPTH HYPERPARAMETER TUNING PROCESS
        
    COULD SATURATION LIMITS/TAPERING ON WEIGHT CHANGES PROVIDE FOR SAFE UPDATES BY LIMITING UPDATE MAGNITUDES?
    COULD BE PARTICULARLY HELPFUL FOR A CORRELATIVE/STATISTICAL GRADIENT APPROACH IN WHICH LARGER GRADIENTS
    MAY NOT HOLD FAR FROM THE UPDATE
    EXPLORE CONDITIONS NEEDED FOR STABLY USING THE SATURATION FACTOR (MAYBE REMOVE NEGATIVE UPDATES OR SEE IF NEGATIVE 
    UPDATES RESULT IN EXPLODING GRADIENTS (FIGURE OUT EXACTLY WHEN EXPLODING GRADIENTS OCCUR, USE SIMPLER FUNCTION PERHAPS
    AND START BY WATCHING MEAN WEIGHT MAGNITUDES), 
    PREVENT UPDATES RESULTING IN NEGATIVE ACTIVATION, CONSIDER RELU ACTIVATION? CONSIDER SIGMOIDAL
    ACTIVATION, SIGMOIDAL AND OTHER SATURATING NON-LINEARITIES MAY MAKE THE BCM SATURATION REDUNDANT
    directly study BCM update function of activation, play with parameters, examine slopes, linear
    regions, negative activity regions, region of instability near singularity, etc.
    PREVENT SIGN CHANGE BY THE SATURATION FACTOR, COULD BE THE CAUSE OF SOME OF THE PROBLEMS
    MOVE TOWARD CORRELATIVE GRADIENT
    test worse BCM rules on smaller problems still
    
    determine size with which BCM rule learns comparably with GD
    then consider distributing error on continuum from BCM to full GD, 
        maybe distribute mean gradient over each layer as error for BCM rule
        maybe distribute error from GD deltas in downstream layer averaged over groups of units
            ISN'T THIS JUST MORE OR LESS A VARIANT OF THE RANDOM FB WEIGHTS?




change linear test to allow for different layer sizes, also need to change learning rule
maybe node perturbation model
extend to non-linearities, need to self.net terms in addition to self.activation terms
MNIST

change tests from functions to objects, allow for more general testing parameters,
such as number of samples, epochs, etc.
add options for various BCM learning rule variants within same BCM object, try to generalize
across datasets

MAKE SURE DIFFERENT TESTS ARE TESTING WITH SAME RANDOM TARGETS

add method for inputting hyperparameters to test objects and to Model class, move more over to parent classes

clean up code, add documentation for all function parameters, add more general comments to structure thoughout
try to reduce redundancy between different models


"""


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

