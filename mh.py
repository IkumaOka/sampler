# 参考：https://qiita.com/amber_kshz/items/3049ab54385e2ce76b29#fnref7
# 提案分布が対象なので、ここで実装してるのはただのメトロポリス法

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
