from itertools import izip
from math import log
import numpy as np
from scipy.misc import logsumexp




class HMM:
    def __init__(self, states, outputs, prob_start=None, trans_prob=None, emit_mu=None, emit_var=None ):
        self.states = states
        self.outputs = outputs
        self.prob_start = prob_start
        self.trans_prob = trans_prob
        #self.emit_prob = normalize(emit_prob, self.states, self.outputs)
        self.emit_mu = emit_mu
        self.emit_var = emit_var

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
            # start_probability
            return []
        
        alpha = [{}]

        # initialize q_0
        for state in self.states:
            alpha[0][state] = self.prob_start[state] * self.emit_prob(state, sequence[0]) 
        # 
        for i in range(len(sequence)):
            alpha.append({})
            for  state_to in self.states:
                prob = 0
                for state_from in self.states:
                    prob += alpha[i][state_from] * self.trans_prob[state_from, state_to]
                alpha[i + 1][state_to] = prob * self.emit_prob(state_to, sequence[i])
        return alpha

    def beta(self, sequence):
        
        if len(sequence) < 1:
            return []

        beta = [{}]

        for state in self.states:
            beta[0][state] = 1

        for i in range(len(sequence)):
            beta.insert(0,{})
            for state_from in self.states:
                prob = 0
                for state_to in self.states:
                    prob += beta[1][state_to]
                    self.trans_prob[state_from, state_to] * self.emit_prob(state_to, sequence[i + 1])

                beta[0][state_from] = prob

        return beta
    
    def likelihood(self, state, output):
        mu = self.emit_mu[state]
        mu_x = mu[0]
        mu_y = mu[1]
        var = self.emit_var[state].reshape((len(output),len(output)))
        # compute multivariate Gaussian
        x = output[0]
        y = output[1]
        var_x = var[0][0]
        var_y = var[1][1]
        pu = (var[0][1]/ np.sqrt(var_x) / np.sqrt(var_y))
        z = ((x - mu_x) ** 2 / var_x) + (( y - mu_y ) ** 2 / var_y) - (2 * pu * ( x - mu_x) * (y - mu_y)/ (var_x * var_y))
        likelihood = 1 / (1 * np.pi * pu * np.sqrt( 1 - pu ** 2)) * np.exp(-1./ 2 * (1 - pu ** 2) * z)
        #likelihood = np.exp(-0.5 * np.transpose(output - mu) * np.linalg.inv(var) * (output - mu)) / (np.sqrt(np.abs(2 * np.pi * var) ))
        return likelihood

    

    def evaluate(self, sequence):
        if len(sequence) < 1:
            return []
        forward = self.alpha(sequence)
        
        # summarize probaly at t = T
        for state in forward[-1]:
            prob += forward[-1][state]
        
        import ipdb; ipdb.set_trace()
        return prob


    def decode(self, sequence): 
        #  Viterbi decoding

        if len(sequence) < 1:
            return []

        delta = {}

        for state in self.states:
            delta[state] = self.prob_start[state] * self.emit_prob(state, sequence[0])

        pre = []

        for i in range(len(sequence)):
            delta_bar = {}
            pre_state = {}
            for state_to in self.states:
                max_prob = 0
                max_state = None
                for state_from in self.states:
                    prob = delta[state_from] * self.trans_prob[state_from, state_to]
                    if prob > max_prob:
                        max_prob = prob
                        max_state = state_from

                delta_bar[state_to] = max_prob * self.emit_prob(state_to, sequence[i])
                pre_state[state_to] = max_state
            delta = delta_bar
            pre.append(pre_state)

        max_state = None
        max_prob = 0

        for state in self.states:
            if delta[state] > max_prob:
                max_prob = delta[state]
                max_state = state
        
        if max_state is None:
            return []

        for i in range(len(sequence)-1, 0, -1):
            max_state = pre[i -1][max_state]
            result.insert(0, max_state)

        return result



    '''def train(seld, sequence, delta=0.000001, smoothing=0):
         # implement EM

         old_likelihood = 0

         for _, outputs in sequences:
             old_likelihood += log(self.evaluate(sequence))

             old_likelihood /= len(sequence)

             while True:
                 update_likelihood = 0
                 for _, outputs in sequences:
                     self.
        ''''




    
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


hmm = HMM(states, outputs, prob_start, trans_prob, emit_mu, emit_var)

hmm.evaluate(train_data)



