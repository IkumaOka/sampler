# 参考サイトURL：https://github.com/amber-kshz/PRML/blob/master/notebooks/Ch11_HMC.ipynb
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

    for i in range(1, L+1):
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

# U(q) = \frac{1}{2} q^2の簡単な式を考える
def grad_U(q):
    return q

q_initial = np.array([0.0])
p_initial = np.array([1.0])

q_exact = np.cos(np.linspace(0, 2*np.pi, 101))
p_exact = np.sin(np.linspace(0, 2*np.pi, 101))

def solve_and_plot_trajectory(ax, q_initial, p_initial, q_exact, p_exact, grad_U, epsilon, L, method):
    q_trajectory, p_trajectory = integrate_eqm(q_initial, p_initial, grad_U, epsilon, L, method=method)
    ax.plot(q_trajectory[:, 0], p_trajectory[:, 0], 'o-')
    ax.plot(q_exact, p_exact, '--', color='k')
    ax.axis("equal")
    ax.set_title(f"{method} : stepsize = {epsilon}")
    plt.savefig("hmc_result/hmc_sampling.png")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

L = 20
for ax, epsilon, method in zip(
    axes.flatten(),
    [0.3, 0.3, 0.3, 1.2, 1.2, 1.2],
    ["euler", "modified_euler", "leapfrog", "euler", "modified_euler", "leapfrog"]
):
    solve_and_plot_trajectory(ax, q_initial, p_initial, q_exact, p_exact, grad_U, epsilon, L, method)

def solve_and_plot_energy(ax, q_initial, p_initial, grad_U, epsilon, L, method):
    q_trajectory, p_trajectory = integrate_eqm(q_initial, p_initial, grad_U, epsilon, L, method=method)
    ax.plot(np.linspace(0, epsilon * L, L + 1), 0.5 * ((q_trajectory**2).sum(axis=1) +  (p_trajectory**2).sum(axis=1)), 'o-', label=f"epsilon={epsilon}")
    E0 = 0.5 * ((q_initial**2).sum() +  (p_initial**2).sum())
    ax.plot([0, epsilon * L], [E0, E0], '--', color='k')

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
for epsilon, L in zip([0.2, 0.4, 0.8], [40, 20, 10]):
    solve_and_plot_energy(ax, q_initial, p_initial, grad_U, epsilon, L, method="leapfrog")

ax.legend()
plt.savefig("hmc_result/hmc_energy.png")