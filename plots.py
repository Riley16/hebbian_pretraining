import numpy as np
import matplotlib.pyplot as plt

def BCM_update_function(post_syn_activity, threshold, Laplace_smoothing_factor):
    num = post_syn_activity * (post_syn_activity - threshold)
    return num / (Laplace_smoothing_factor + num)

def simple_BCM_update_function(post_syn_activity, threshold, Laplace_smoothing_factor):
    num = (post_syn_activity - threshold)
    return num / (Laplace_smoothing_factor + np.maximum(num, 0))


threshold = 1 # 1
threshold_simple = 1 # 1

smoothing_factor = 0.5 # 0.5
smoothing_factor_simple = 0.5 # 0.5


n_points = 4000 # 4000
# post_syn_activity = np.array([0.001 * i for i in range(-n_points, n_points)])
post_syn_activity = np.array([0.001 * i for i in range(round(0*n_points), round(1*n_points))])
plt.plot(post_syn_activity, BCM_update_function(post_syn_activity, threshold, smoothing_factor),  label="Full BCM")
plt.plot(post_syn_activity, simple_BCM_update_function(post_syn_activity, threshold_simple, smoothing_factor_simple), label="Simplified initial model")
plt.legend()
plt.grid(axis='y')
# plt.title(r'$ s_{5} $')


plt.show()



