from itertools import izip
from math import log
import numpy as np
from scipy.misc import logsumexp




class HMM:
    def __init__(self, states, outputs, prob_start=None, trans_prob=None, emit_prob=None ):
        self.states = states
        self.symbols = symbols
        self.prob_start = normalize(prob_start, self.states)
        self.trans_prob = normalize(trans_prob, self.states, self.outputs)
        self.emit_prob = normalize(emit_prob, self.states, self.outputs)
        
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
        
        if (state not in self.states) or (output not in self.outputs):
            return 0

        return self.emit_prob[state][output]

    def alpha(self, seuqence):
        if len(sequence) == 0:
            # start_probability
            return []
        
        alpha = [{}]

        # initialize q_0
        for state in self.states:
            alpha[0][state] = self.prob_start(state) * self.emit_prob(state, sequence[0]) 

        # 
        for i in range(len(sequence)):
            alpha.append({})
            for  state_to in self.states:
                prob = 0
                for state_from in self.states:
                    prob += alpha[i][state_from] * self.trans_prob(state_from, state_to)
                    
                alpha[i + 1][state_to] = prob * self.emit_prob(state_to, sequence[index])

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
                    self.trans_prob(state_from, state_to) * self.emit_prob(state_to, sequence[i + 1])

                beta[0][state_from] = prob

        return beta
    
    def likelihood(self, state, output):
        mu = self.emit_prob[state]
        var = self.emit_var[state]
        
        likelihood = 



    def inference(self, sequence):
        
        if len(sequence) < 1:
            return 0

        prob = 0
        forward = self.alpha(sequence)
        for state in forward[-1]:
            prob += forward[-1][state]

        return prob
        


    
def load_file(filename):
    with open(filename, 'r') as input_file:
        data = np.load(input_file)
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

train_data = load_file('hwk3data/EMGaussian.train')
test_data = load_file('hwk3data/EMGaussian.test')


prob_start = init_startP(K)
trans_prob = init_trans(K)

emit_mu = np.asarray([[-2.0344, 4.1726],[3.9779, 3.7735],[3.9007, -3.7972],[-3.0620, -3.5345]])
emit_var = np.asarray([[2.9044, 0.2066, 0.2066, 2.7562], [0.2104, 0.2904, 0.2904, 12.2392], [0.9213, 0.0574, 0.0574, 1.8660], [6.2414, 6.0502, 6.0502, 6.1825]])









