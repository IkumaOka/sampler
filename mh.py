# 参考：https://qiita.com/amber_kshz/items/3049ab54385e2ce76b29#fnref7
# 提案分布が対象なので、ここで実装してるのはただのメトロポリス法


import numpy as np
import random 
from matplotlib import pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


class MetropolisSampler:
    def __init__(self, D, sigma, p):
        '''
        D: データXの次元
        sigma: 提案分布の共分散行列
        p: p^{~}のこと。サンプルを取りたい確率密度関数（正規化されているとは限らない）
        '''
        self.D = D
        self.sigma = sigma
        self.p = p

    # 受理確率を計算 PRML(11.33)
    def _A(self, x, y):
        denom = self.p(y)
        if denom == 0:
            return 1.0
        else:
            return min(1.0, self.p(x) / denom)
    
    def sample(self, init_x, N, stride):
        X = np.zeros((N, self.D))
        x = init_x
        cnt = 0
        for i in range((N-1)*stride+1):
            tmp_x = np.random.multivariate_normal(x, self.sigma)
            tmp_rand = random.random()
            if tmp_rand < self._A(tmp_x, x):
                x = tmp_x
            if i % stride == 0:
                X[cnt] = x
                cnt += 1
        return X

# 1次元ガンマ分布
def p_gamma(x):
    if x > 0:
        return (x**2) * np.exp(-x)
    else:
        return 0.0

sampler = MetropolisSampler(D=1, sigma=np.array([[1.0]]), p = p_gamma)
samples = sampler.sample(init_x=np.array([1.0]), N=5000, stride=10)[:, 0]

print(f"sample mean : {np.mean(samples)}")
print(f"sample variance : {np.var(samples)}")
step = 0.25
fig = plt.figure()
plt.hist(samples, bins=np.arange(0, 10, step), density=True, label="Normalized histogram")
xx = np.linspace(0,10,100)
plt.plot(xx, 0.5*xx**2*np.exp(-xx), label="probability density function")
plt.legend()
fig.savefig("gamma_dist_sampling.png")


def p(x):
    precmat = np.array([[10,-6],[-6,10]])
    return np.exp(-0.5*x @ precmat @ x)

sampler = MetropolisSampler(D = 2, sigma = np.array([[1,0],[0,1]]), p = p)
samples = sampler.sample(np.array([1,0]), N=5000, stride=10)
fig = plt.figure(figsize=(12,5))

print(f"sample mean : {np.mean(samples, axis=0)}")
print(f"sample covariance : {np.cov(samples, rowvar=False)}")

# two dimensional histogram
H, xx, yy = np.histogram2d(samples[:,0], samples[:,1], bins=25, normed=True)
H = H.T
XX, YY = np.meshgrid(xx, yy)

# contour plot of the density function
xx_f = np.linspace(-1.5,1.5,100)
yy_f = np.linspace(-1.5,1.5,101)
XX_f, YY_f = np.meshgrid(xx_f, yy_f)
Z_f = np.exp( -0.5*( 10*XX_f*XX_f -12*XX_f*YY_f + 10*YY_f*YY_f ))/(2*np.pi)*8


plt.subplot(121)
plt.plot(samples[:,0], samples[:,1],',')
plt.contour(XX_f,YY_f,Z_f)
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(XX, YY, H)
plt.colorbar()
plt.contour(XX_f,YY_f,Z_f)
fig.savefig("2D_normal_sampling.png")