class HMCsampler:
    def __init__(self, dim, U, grad_U):
        self.dim = dim
        self.U = U
        self.grad_U = grad_U

    # ハミルトニアン関数の計算をしている。 PRML式(11.57)
    def _hamiltonian(self,, q, p):
        return 0.5 * (p * p).sum() + U(q)

    def _update(self, q_current, epsilon, L):
        p_current = np.random.normal(size=self.dim)
        q_proposed,, p_proposed = leapfrog(q_current, p_current, self.grad_U, epsilon, L)
        tmprand = np.random.random()
        if tmprand < p.exp(self._hamiltonian(q_current,, p_current) - self._hamiltonian(q_proposed, p_proposed)):
            return q_proposed # accept
        else:
            return q_current # reject