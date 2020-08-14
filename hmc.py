# 参照サイト：https://nbviewer.jupyter.org/github/amber-kshz/PRML/blob/master/notebooks/Ch11_HMC.ipynb
import numpy as np
import random 
from matplotlib import pyplot as plt

class HMCsampler:
    def __init__(self, dim, U, grad_U):
        self.dim = dim
        self.U = U
        self.grad_U = grad_U

    # ハミルトニアン関数の計算をしている。 PRML式(11.57)
    def _hamiltonian(self, q, p):
        return 0.5 * (p * p).sum() + U(q)

    def _update(self, q_current, epsilon, L):
        p_current = np.random.normal(size=self.dim)
        q_proposed, p_proposed = leapfrog(q_current, p_current, self.grad_U, epsilon, L)
        tmprand = np.random.random()
        if tmprand < np.exp(self._hamiltonian(q_current, p_current) - self._hamiltonian(q_proposed, p_proposed)):
            return q_proposed # accept
        else:
            return q_current # reject
    
    def sample(self, q_initial, n_samples, epsilon, L):
        Q = np.zeros((n_samples, self.dim))
        q_current = q_initial
        for i in range(n_samples):
            q_current = self._update(q_current, epsilon, L)
            Q[i] = q_current
        return Q

def leapfrog(q_initial, p_initial, grad_U, epsilon, L):
    q = q_initial
    p = p_initial
    
    p = p - epsilon / 2 * grad_U(q)  
    for i in range(1, L):
        q = q + epsilon * p  
        p = p - epsilon * grad_U(q)  
    q = q + epsilon * p  
    p = p - epsilon / 2 * grad_U(q)  
    return q, p

def U(q):
    return 0.5 * ((q**2).sum())

def grad_U(q):
    return q

hmc_1dgaussian = HMCsampler(dim=1, U=U, grad_U=grad_U)

def sample_hmc_1dgaussian(hmc, q_initial, n_samples, epsilon, L, idx_start=0):
    # get the samples
    samples = hmc.sample(
        q_initial=q_initial,
        n_samples=n_samples,
        epsilon=epsilon,
        L=L
    )
    print(samples.shape)
    print(f"sample mean : ", np.mean(samples[idx_start:, 0]))
    print(f"sample variance : ", np.var(samples[idx_start:, 0]))

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    # plot histogram
    ax = axes[0]
    ax.hist(samples, bins=np.arange(-3, 3, 0.2), density=True, label="Normalized histogram")
    xx = np.linspace(-3, 3, 100)
    ax.plot(xx, np.exp(-0.5*xx*xx)/(np.sqrt(2*np.pi)), label="probability density function")
    # plot time series of q
    axes[1].plot(samples[:, 0], '.')
    plt.savefig("hmc_result/hmc_1dgaussian_sampling.png")

sample_hmc_1dgaussian(hmc_1dgaussian, q_initial=np.array([1.0]), n_samples=500, epsilon=0.3, L=13, idx_start=10)

