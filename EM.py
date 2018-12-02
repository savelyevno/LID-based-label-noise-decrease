import numpy as np
import matplotlib.pyplot as plt


def initialize(n_dst, data, mu_0):
    # initialize the mu_0 randomly
    dim = data.shape[1]
    if mu_0 is None:
        mu_0 = np.zeros((n_dst, dim))

    for row in range(n_dst):
        idx = np.random.randint(data.shape[0])
        for col in range(dim):
            mu_0[row][col] += data[idx][col]

    # initialize the sigma_0
    sigma_0 = []
    for k in range(n_dst):
        sigma = np.cov(data.T)
        if sigma.shape == ():
            sigma = np.array([[sigma]])
        sigma_0.append(sigma)
        # sigma_0.append(np.mat(np.random.random((dim,dim))))

    # initialize the pi randomly
    sum_pi = 1.0
    pi = np.zeros(n_dst)
    pi += sum_pi/n_dst

    return mu_0, sigma_0, pi


def e_step(n_dst, mu, sigma, pi, data):
    """
    Evaluate the responsibility (prediction confidence) using the current parameter values
    """
    N = data.shape[0]

    conf = np.zeros((N, n_dst))

    for i in range(N):
        for k in range(n_dst):
            conf[i][k] = (pi[k] * gdf(data[i], mu[k], sigma[k])) / prior_prob(n_dst, mu, sigma, pi, data[i])

    return conf


def m_step(conf, n_dst, data):
    """
    Re-estimate the parameters using the current
    responsibility
    """
    # calculate new_mu_k
    N = data.shape[0]
    dim = data.shape[1]
    n_per_dst = np.zeros(n_dst)

    new_mu_k = np.zeros((n_dst, dim))
    for k in range(n_dst):
        for n in range(N):
            n_per_dst[k] += conf[n][k]
            new_mu_k[k] += (conf[n][k] * data[n])

        new_mu_k[k] /= n_per_dst[k]

    new_sigma_k = np.zeros((n_dst, dim, dim))
    for k in range(n_dst):
        for n in range(N):
            x_n = np.zeros((1, dim))
            mu_n = np.zeros((1, dim))
            x_n += data[n]
            mu_n += new_mu_k[k]
            x_mu = x_n - mu_n
            new_sigma_k[k] += (conf[n][k] * x_mu * x_mu.T)
        new_sigma_k[k] /= n_per_dst[k]

    # calculate new_pi_k
    new_pi_k = np.zeros(n_dst)
    for k in range(n_dst):
        new_pi_k[k] += (n_per_dst[k]/N)

    return new_mu_k, new_sigma_k, new_pi_k


def likelihood(n_dst, mu, sigma, pi, data):
    """
    Calculate the log likelihood using current mu, sigma, and
    pi.
    """
    log_score = 0.0

    for n in range(len(data)):
        log_score += np.log(prior_prob(n_dst, mu, sigma, pi, data[n]))
    return log_score


def prior_prob(n_dst, mu, sigma, pi, data):
    pb = 0.0
    for k in range(n_dst):
        pb += pi[k] * gdf(data, mu[k], sigma[k])

    return pb


def gdf(x, mu, sigma):
    x_mu = np.matrix(x - mu)
    inv_sigma = np.linalg.inv(sigma)
    det_sqrt = np.linalg.det(sigma)**0.5

    norm_const = 1 / ((2 * np.pi)**(len(x) / 2) * det_sqrt)
    exp_value = np.power(np.e, -0.5 * (x_mu * inv_sigma * x_mu.T))
    score = norm_const * exp_value

    return score


def em_solve(n_dst, data, min_iter=15, max_iter=50, threshold=1e-4, mu_0=None, verbose=0):
    mu, sigma, pi = initialize(n_dst, data, mu_0)
    # epsilon = 1e-30
    log_score = likelihood(n_dst, mu, sigma, pi, data)
    # log_score_0 = log_score
    i = 0
    while i < max_iter:
        # expectation step
        conf = e_step(n_dst, mu, sigma, pi, data)

        # maximization step
        mu, sigma, pi = m_step(conf, n_dst, data)

        # evaluate the log likelihood
        new_log_score = likelihood(n_dst, mu, sigma, pi, data)
        log_score_diff = abs(new_log_score - log_score)
        if log_score_diff < threshold and i > min_iter:
            # print abs(new_log_score - log_score)
            break

        log_score = new_log_score

        i += 1
        if verbose == 1:
            print('iteration: %d; log score diff: %g' % (i, log_score_diff[0, 0]))

    return mu, sigma, pi, conf


def test_em():
    np.random.seed(0)
    # dim = 2
    N1 = 200
    mu1 = np.array([0, 0])
    sigma1 = np.array([[1, 0], [0, 1]])
    x1 = np.random.multivariate_normal(mu1, sigma1, N1)

    N2 = 200
    mu2 = np.array([3, 3])
    sigma2 = np.array([[2, 1], [1, 3]])
    x2 = np.random.multivariate_normal(mu2, sigma2, N2)

    data = np.append(x1, x2, 0)
    np.random.shuffle(data)

    mus, sigmas, pi, conf = em_solve(2, data, 50, verbose=1)

    for i in range(len(mus)):
        print('mu', i, mus[i])

    for i in range(len(sigmas)):
        print('sigma', i, '\n', sigmas[i])

    N = data.shape[0]

    for i in range(N):
        plt.plot(data[i][0], data[i][1], 'o', color='C' + str(np.argmax(conf[i])))

    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_em()
