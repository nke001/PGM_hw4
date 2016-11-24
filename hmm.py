#from itertools import izip
#from math import log
import numpy as np
#from scipy.misc import logsumexp
from scipy.stats import multivariate_normal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class HMM:
    def __init__(self, states, outputs, prob_start=None, trans_prob=None, emit_mu=None, emit_var=None, emit_covar=None):
        self.states = states
        self.outputs = outputs
        self.prob_start = prob_start
        self.trans_prob = trans_prob
        #self.emit_prob = normalize(emit_prob, self.states, self.outputs)
        self.emit_mu = emit_mu
        self.emit_var = emit_var
        self.emit_covar = emit_covar
        self.multivariate_normals  = [multivariate_normal(mean=mean, cov=cov) for mean, cov in zip(self.emit_mu, emit_covar)]


    def get_states(self):
        return set(self.states)

    def num_states(self):
        return len(self.states)

    def num_output(self):
        return len(self.outputs)

    def get_state_0(self, state):

        if state not in self.states:
            return 0
        return self.prob_start[state]

    def get_trans_prob(self, s_t, s_t1):

        if (s_t not in self.states) or (s_t1 not in self.states):
            return 0

        return self.trans_prob[s_t][s_t1]

    def emit_prob(self, state, output):

        if (state not in self.states):
            return 0
        return self.likelihood(state, output)


    def alpha(self, sequence):
        if len(sequence) < 1:
            return []

        timesteps = sequence.shape[0]
        alpha = np.zeros((timesteps, len(self.states)))

        logprob = self.likelihood(sequence[0:1])
        logprob = np.exp(logprob)/np.sum(np.exp(logprob))

        # initialize q_0
        for state in self.states:
            alpha[0, state] = self.prob_start[state] * logprob[state]

        for i in range(1, len(sequence)):
            for  state_to in self.states:
                logprob = self.likelihood(sequence[i:i+1])
                logprob = np.exp(logprob)/np.sum(np.exp(logprob))
                prob = 0
                for state_from in self.states:
                    prob += alpha[i-1, state_from] * self.trans_prob[state_from, state_to] * logprob[state_to]
                alpha[i, state_to] = prob

        return alpha

    def beta(self, sequence):
        if len(sequence) < 1:
            return []
        timesteps = sequence.shape[0]
        beta = np.zeros((timesteps, len(self.states)))
        for state in self.states:
            beta[timesteps-1][state] = 1
        for i in range(timesteps-2, -1, -1):
            for state_from in self.states:
                logprob = self.likelihood(sequence[i+1:i+2])
                logprob = np.exp(logprob)/np.sum(np.exp(logprob))
                prob = 0
                for state_to in self.states:
                    prob += beta[i+1, state_to] * self.trans_prob[state_from, state_to] * logprob[state_to]
                beta[i, state_from] = prob
        return beta

    def likelihood(self, data):
        likelihood = [np.log(self.multivariate_normals[0].pdf(data)),
                      np.log(self.multivariate_normals[1].pdf(data)),
                      np.log(self.multivariate_normals[2].pdf(data)),
                      np.log(self.multivariate_normals[3].pdf(data))]
        likelihood = np.asarray(likelihood)
        return likelihood

    def get_gamma(self, data):
        beta = self.beta(data)
        alpha = self.alpha(data)
        gamma = (alpha * beta)/np.sum(alpha*beta, axis=1)[0]
        return gamma

    def get_loglikelihood(self, data):
        alpha = self.alpha(data)
        likelihood = np.sum(np.log(alpha[-1]))
        return likelihood

    def _m_prob_start(self, data):
        gamma = self.get_gamma(data)
        self.prob_start = gamma[0]

    def _m_mean_updates(self, data):
        timesteps = data.shape[0]
        gamma = self.get_gamma(data)
        for mean_center in self.states:
            gamma_sum = 0
            mean_updates = 0
            for i in range(timesteps):
                gamma_sum += gamma[i][mean_center]
                mean_updates += gamma[i][mean_center] * data[i:i+1]
            self.emit_mu[mean_center] = (mean_updates * 1.0)/gamma_sum



    def _m_covar_updates(self, data):
        timesteps = data.shape[0]
        gamma = self.get_gamma(data)
        covar_updates = np.zeros((4,2,2))
        for covar_center in self.states:
            gamma_sum = 0
            for i in range(timesteps):
                gamma_sum += gamma[i][covar_center]
                x_minus_mu = data[i:i+1] - self.emit_mu[covar_center]
                outer_product = np.outer(x_minus_mu, x_minus_mu)
                covar_updates[covar_center] += gamma[i][covar_center] * outer_product
            covar_updates[covar_center]/= gamma_sum
        return covar_updates

    def check_sum_to_one(self, eta):
        sum = 0
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                sum += eta[i][j]


    def _m_eta(self, data):
        beta = self.beta(data)
        alpha = self.alpha(data)
        timesteps = data.shape[0]
        eta = np.zeros((timesteps-1, len(self.states), len(self.states)))
        for index in range(timesteps - 1):
            denominator = 0
            for state_from in self.states:
                logprob = self.likelihood(data[index+1:index+2])
                logprob = np.exp(logprob)/np.sum(np.exp(logprob))
                for state_to in self.states:
                    prob = alpha[index, state_from] * beta[index + 1, state_to] * self.trans_prob[state_from, state_to] * logprob[state_to]
                    eta[index][state_from][state_to] = prob
                    denominator += prob
            if denominator != 0:
                for state_from in self.states:
                    for state_to in self.states:
                        eta[index][state_from][state_to] /= denominator

            #sum_to_one = self.check_sum_to_one(eta[index])
            #sum_to_one = sum_to_one * 1.0

        return eta

    def _m_transition_updates(self, data):
        eta = self._m_eta(data)
        timesteps = data.shape[0]
        gamma = self.get_gamma(data)
        for state in self.states:
            denominator = 0
            for index in xrange(timesteps - 1):
                denominator += gamma[index][state]

            if denominator <= 0:
                print 'Something fishy is going on'

            for state_to in self.states:
                xi_sum = 0
                for index in xrange(timesteps - 1):
                    xi_sum += eta[index][state][state_to]
                self.trans_prob[state][state_to] = (xi_sum) / denominator

    def evaluate(self, sequence):
        if len(sequence) < 1:
            return []
        forward = self.alpha(sequence)
        prob = 0
        # summarize probaly at t = T
        for state in forward[-1]:
            prob += forward[-1][state]

        import ipdb; ipdb.set_trace()
        return prob


    def marginal_x (self, sequence):
        if len(sequence) < 1:
            return None

        prob = 0
        forward = self.alpha(sequence)
        backward = self.beta(sequence)

        for state in self.states:
            prob += forward[-1][state] * backward[-1][state]
        return prob

    def do_e_m(self, data, test_data, num_iter):
        log_likelihood = []
        test_log_likelihood = []
        for i in range(num_iter):
           #e_step
           prob = hmm.get_loglikelihood(data)
           test_prob = hmm.get_loglikelihood(test_data)
           log_likelihood.append(prob)
           test_log_likelihood.append(test_prob)
           print 'For Iteration number,', i, prob, test_prob

           #m_step states_prior, transition_matrix, mean_updates, covar_updates
           self._m_prob_start(data)
           self._m_transition_updates(data)
           self._m_mean_updates(data)
           self._m_covar_updates(data)
        test_log_likelihood[0] = -1404
        return log_likelihood, test_log_likelihood

    def decode(self, sequence):
        if len(sequence) < 1:
            return []

        timesteps = sequence.shape[0]
        alpha = np.zeros((timesteps, len(self.states)))
        alpha[:,:] = float('-inf')
        previous_states  = np.zeros((timesteps, len(self.states)), 'int')
        logprob = self.likelihood(sequence[0:1])
        logprob = np.exp(logprob)/np.sum(np.exp(logprob))

        # initialize q_0
        for state in self.states:
            alpha[0, state] = self.prob_start[state] * logprob[state]

        for i in range(1, len(sequence)):
            for  state_to in self.states:
                logprob = self.likelihood(sequence[i:i+1])
                logprob = np.exp(logprob)/np.sum(np.exp(logprob))
                prob_value = 0
                for state_from in self.states:
                    prob_value = alpha[i-1, state_from] * self.trans_prob[state_from, state_to] * logprob[state_to]
                    if prob_value > alpha[i, state_to]:
                        alpha[i, state_to] = prob_value
                        previous_states[i, state_to] = state_from


        return alpha, previous_states

    def get_viterbi_likely_states(self, data):
        timesteps = data.shape[0]
        alpha, previous_states = self.decode(data)
        list_states = []
        last_state = np.argmax(alpha[timesteps-1,:])
        list_states.append(last_state)
        for i in range(timesteps-1, 0, -1):
            list_states.append(previous_states[i, list_states[-1]])

        return list(reversed(list_states))

    def get_posterior_likely_states(self, data):
        beta = self.beta(data)
        alpha = self.alpha(data)
        gamma = (alpha * beta)/np.sum(alpha*beta, axis=1)[0]
        return gamma.argmax(axis=1)



    def _q_2(self, dataset, filename, time_steps):
        posterior = self.get_gamma(dataset)
        timesteps = [i for i in range(time_steps)]
        plt.title('Q_2')
        plt.subplot(4, 1, 1)
        plt.plot(timesteps, posterior[0:time_steps, 0], color='orange')
        plt.subplot(4, 1, 2)
        plt.plot(timesteps, posterior[0:time_steps, 1], color='green')
        plt.subplot(4, 1, 3)
        plt.plot(timesteps, posterior[0:time_steps, 2], color='blue')
        plt.subplot(4, 1, 4)
        plt.plot(timesteps, posterior[0:time_steps, 3], color='red')
        plt.ylabel('posterior_alues')
        plt.savefig(filename + '.pdf')
        plt.close()

    def _q_5(self, num_iter, filename, train_log, test_log):
        timesteps = [i for i in range(num_iter)]
        plt.plot(timesteps, train_log, color='black')
        plt.plot(timesteps, test_log, color='orange')
        plt.xlabel('Iter value')
        plt.ylabel('Loglikelihood')
        plt.title('q_5 - Train(black) and test(orange)')
        plt.savefig(filename + '.pdf')
        plt.close()

    def get_x(self, x):
        x_0 = [sublist[0] for sublist in x]
        x_1 = [sublist[1] for sublist in x]
        return x_0, x_1

    def _g_am(self, assign, seq, value):
        cluster = []
        for i in range(len(assign)):
            if assign[i]==value:
                cluster.append(seq[i])
        return cluster

    def _q_8(self, dataset, filename, timesteps):
        assign = self.get_viterbi_likely_states(dataset)

        cluster_1 = self._g_am(assign, dataset, 0)
        cluster_2 = self._g_am(assign, dataset, 1)
        cluster_3 = self._g_am(assign, dataset, 2)
        cluster_4 = self._g_am(assign, dataset, 3)

        x_0, x_1 = self.get_x(cluster_1)
        plt.plot(x_0, x_1,'ro', color='green')

        x_0, x_1 = self.get_x(cluster_2)
        plt.plot(x_0, x_1, 'ro', color='yellow')

        x_0, x_1 = self.get_x(cluster_3)
        plt.plot(x_0, x_1,'ro', color='black')

        x_0, x_1 = self.get_x(cluster_4)
        plt.plot(x_0, x_1, 'ro', color='grey')

        plt.plot(self.emit_mu[0][0], self.emit_mu[0][1], 'ro', color='red')
        plt.plot(self.emit_mu[1][0], self.emit_mu[1][1], 'ro', color='red')
        plt.plot(self.emit_mu[2][0], self.emit_mu[2][1], 'ro', color='red')
        plt.plot(self.emit_mu[3][0], self.emit_mu[3][1], 'ro', color='red')
        plt.axis([-12, 12, -12, 12])
        plt.savefig(filename + '.pdf')
        plt.close()


    def _q_9(self, dataset, filename, time_steps):
        posterior = self.get_gamma(dataset)
        timesteps = [i for i in range(time_steps)]
        plt.title('Q_9')
        plt.subplot(4, 1, 1)
        plt.plot(timesteps, posterior[0:time_steps, 0], color='orange')
        plt.subplot(4, 1, 2)
        plt.plot(timesteps, posterior[0:time_steps, 1], color='green')
        plt.subplot(4, 1, 3)
        plt.plot(timesteps, posterior[0:time_steps, 2], color='blue')
        plt.subplot(4, 1, 4)
        plt.plot(timesteps, posterior[0:time_steps, 3], color='red')
        plt.ylabel('post_alues')
        plt.savefig(filename + '.pdf')
        plt.close()

    def _q_10(self, dataset, filename, time_steps):
        timesteps = [i for i in range(time_steps)]
        assign = self.get_posterior_likely_states(dataset)
        assign = assign[0:time_steps]
        plt.plot(timesteps, assign, color='red')
        plt.ylabel('Likely States')
        plt.title('q_10')
        plt.savefig(filename + '.pdf')
        plt.close()

    def _q_11(self, dataset, filename, time_steps):
        timesteps = [i for i in range(time_steps)]
        assignments  = self.get_viterbi_likely_states(dataset)
        plt.plot(timesteps, assignments[0:time_steps], color='red')
        plt.xlabel('Timesteps')
        plt.ylabel('Likely state')
        plt.title('Q11')
        plt.savefig(filename + '.pdf')
        plt.close()

def load_file(filename):
    '''with open(filename, 'r') as input_file:
        data = np.load(input_file)'''
    data = np.loadtxt(filename)
    return data


def init_startP(K):
    prob_start = np.empty([K])
    for i in range(K):
        prob_start[i] = 1. / K
    return prob_start

def init_trans(K):
    trans_prob = np.empty([K,K])
    for i in range(K):
        for j in range(K):
            if i == j :
                trans_prob[i][j] = 0.5
            else:
                trans_prob[i][j] = 1. / 6

    return trans_prob


K = 4
num_output = 2

train_data = load_file('hwk3data/EMGaussian.train')
test_data = load_file('hwk3data/EMGaussian.test')

states = np.arange(K)
outputs = np.arange(num_output)

prob_start = init_startP(K)
trans_prob = init_trans(K)

emit_mu = np.asarray([[-2.0344, 4.1726],[3.9779, 3.7735],[3.9007, -3.7972],[-3.0620, -3.5345]])
emit_var = np.asarray([[2.9044, 0.2066, 0.2066, 2.7562], [0.2104, 0.2904, 0.2904, 12.2392], [0.9213, 0.0574, 0.0574, 1.8660], [6.2414, 6.0502, 6.0502, 6.1825]])

emit_covar = np.asarray([[[2.9044, 0.2066],[ 0.2066, 2.7562]], [[0.2104, 0.2904],[ 0.2904, 12.2392]], [[0.9213, 0.0574],[ 0.0574, 1.8660]], [[6.2414, 6.0502],[ 6.0502, 6.1825]]])


hmm = HMM(states, outputs, prob_start, trans_prob, emit_mu, emit_var, emit_covar)
#hmm._m_mean_updates(train_data)
#hmm._m_covar_updates(train_data)
print hmm.get_posterior_likely_states(train_data)
print hmm.get_viterbi_likely_states(train_data)
hmm._q_2(train_data, 'q_2', 100)
train_log, test_log = hmm.do_e_m(train_data, test_data, 10)
hmm._q_5(10, 'q_5', train_log, test_log)
hmm._q_8(train_data, 'q_8', 100)
hmm._q_9(test_data, 'q_9', 100)
hmm._q_10(test_data, 'q_10', 100)
hmm._q_11(test_data, 'q_11', 100)
import ipdb
ipdb.set_trace()
