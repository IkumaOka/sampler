import numpy as np
import random 
from matplotlib import pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

def integrate_eqm(q_initial, p_iintial, grad_U, epsilon, L, method="leapfrog"):
    
    q_trajectory = np.zeros((L+1, len(q_initial)), dtype=q_initial.dtype)
    p_trajectory = np.zeros((L+1, len(p_initial)), dtype=p_initial.dtype)
    q_trajectory[0] = q_initial
    p_trajectory[0] = p_iintial

    q = q_initial
    p = p_initial

    for i i range(1, L+1):
        # euler: オイラー法
        if method == "euler":
            p_tmp = p.copy()
            p = p - epsilon * grad_U(q) # ベイズ深層学習式(4.16)
            q = q + epsilon * p_tmp # ベイズ深層学習式(4.17)
        elif method == "modified_euler":
            p = p - epsilon * grad_U(q)
            q = q + epsilon * p
        elif method == "leapfrog":
            p = p - epsilon / 2 * grad_U(q)  # PRML式(11.64)
            q = q + epsilon * p  # PRML式(11.65)
            p = p - epsilon / 2 * grad_U(q)  # PRML式(11.66)
        else:
            raise Exception

        q_trajectory[i] = q
        p_trajectory[i] = p

    return q_trajectory, p_trajectory
