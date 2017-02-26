########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang, Avishek Dutta
# Description:  Set 5 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (i.e. run `python 2G.py`) to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np
import time

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.
            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            D:          Number of observations.
            A:          The transition matrix.
            O:          The observation matrix.
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''
        self.L = len(A)
        self.D = len(O[0])
        self.A = np.array(A)
        self.O = np.array(O)
        self.A_start = np.array([1. / self.L for i in range(self.L)])


    def normal(self, array):
        return ([float(i)/sum(array) for i in array])

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    Output sequence corresponding to x with the highest
                        probability.
        '''
        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = np.zeros((M + 1, self.L))
        seqs = np.zeros((M + 1, self.L), 'int')

        # initial guess
        probs[1,:] = self.A_start * self.O[:,x[0]]

        # go through rest of the input states
        for i in range(2, M + 1):
            for s in range(self.L):
                prob_choices = probs[i-1,:]*self.A[:, s]*self.O[s, x[i - 1]]

                # pick the highest probability one
                best = np.argmax(prob_choices)
                probs[i, s] = prob_choices[best]
                seqs[i, s] = best

        # pick the best sequence (back-tracking)
        max_seq = str(np.argmax(probs[-1:,]))
        for i in range(2, M + 1)[::-1]:
            max_seq = str(seqs[i, int(max_seq[0])]) + max_seq
        return(max_seq)

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''
        M = len(x)      # Length of sequence.
        alphas = np.zeros((M + 1, self.L))

        # initialize
        alphas[1,:] = self.A_start * self.O[:,x[0]]
        if (normalize==True):
            alphas[1,:] = self.normal(alphas[1,:])

        # go through the rest of the input states
        for i in range(2, M + 1):
            for s in range(self.L):
                alphas[i][s] = self.O[s, x[i-1]]*np.dot(alphas[i - 1,:], self.A[:, s])
            if (normalize==True):
                alphas[i,:] = self.normal(alphas[i,:])

        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
        '''
        M = len(x)      # Length of sequence.
        betas = [[0. for i in range(self.L)] for j in range(M + 1)]

        # initialize
        betas = np.zeros((M + 1, self.L))
        betas[M,:] = 1*np.ones(self.L)
        if (normalize==True):
            betas[M,:] = self.normal(betas[M,:])

        # go through the rest of the input states
        for i in range(M - 1, 0, -1):
            for s in range(self.L):
                subsum = 0
                for t in range(self.L):
                    subsum += betas[i+1, t]*self.A[s, t]*self.O[t, x[i]]
                betas[i][s] = subsum
            if (normalize==True):
                betas[i,:] = self.normal(betas[i,:])

        # first row (have to incorporate A_start)
        for s in range(self.L):
            subsum = 0
            for t in range(self.L):
                subsum += betas[1][t]*self.A_start[t]*self.O[t,x[0]]
            betas[0][s] = subsum
        if (normalize==True):
            betas[0,:] = self.normal(betas[0,:])

        return(betas)

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''
        # Calculate each element of A using the M-step formulas.
        (n, m) = self.A.shape
        A_improved = np.zeros((n,m))
        denom =0
        for a in range(n):
            for b in range(m):
                numerator = 0
                denom = 0
                for y_sub in Y:
                    for i in range(len(y_sub)-1):
                        if (y_sub[i] == b):
                            denom += 1
                        if (y_sub[i] == b and y_sub[i+1] == a):
                            numerator += 1
                #denom = sum(Y, []).count(b)
                A_improved[b,a] = (float(numerator)/denom)
        self.A = A_improved

        # Calculate each element of O using the M-step formulas.
        (n, m) = self.O.shape
        O_improved = np.zeros((n,m))

        for z in range(n):
            for w in range(m):
                numerator = 0
                denom = 0
                for i in range(len(Y)):
                    y_sub = Y[i]
                    x_sub = X[i]
                    for j in range(len(y_sub)):
                        if (x_sub[j] == w and y_sub[j] == z):
                            numerator += 1
                    denom += y_sub.count(z)
                O_improved[z, w] = float(numerator)/denom
        self.O = O_improved

        pass

    def unsupervised_learning(self, X):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.
        '''

        # calculate (6) in HMM notes
        def calcGammaTable(alpha, beta, length):
            output = np.zeros((self.L, length),float)
            for j in range(length):
                numerator = alpha[j+1,:]*beta[j+1,:]
                denom = np.dot(alpha[j+1,:],beta[j+1,:])
                output[:,j] = numerator/denom
            return(output)

        # calculates (7) in HMM notes
        def calcIotaTable(alpha, beta, x):
            output = np.zeros((self.L, self.L, len(x)-1),float)
            output2 = np.zeros((len(x)-1), float)
            for j in range(len(x)-1):
                denom = 0
                for a in range(self.L):
                    for b in range(self.L):
                        output[a,b,j] = alpha[j+1,a]*beta[j+2,b]*self.A[a,b]*self.O[b,x[j+1]]
                        denom += output[a,b,j]
                output2[j] = denom
            return(output, output2)

        # calculates (14) in HMM notes
        def newTransition(a, b, sub_x, alpha, beta):
            numerator = sum(self.iota_output[a,b,:]/self.iota_denom[:])
            return(numerator)

        # calculates (15) in HMM notes
        def newEmission(a, w, sub_x, alpha, beta):
            indices = np.where(np.array(sub_x) == w)[0]
            numerator = sum([self.gamma_output[a,j] for j in indices])
            return (numerator)

        # initialize
        A_top = np.zeros((self.L,self.L),float)
        A_bot = np.zeros((self.L,self.L),float)
        O_top = np.zeros((self.L,self.D),float)
        O_bot = np.zeros((self.L,self.D),float)

        # run through all the half-year dates
        for num in range(len(X)):
            sub_x = X[num]
            alpha = self.forward(sub_x, True)
            beta = self.backward(sub_x, True)
            self.gamma_output = calcGammaTable(alpha, beta, len(sub_x))
            (self.iota_output, self.iota_denom) = calcIotaTable(alpha, beta, sub_x)

            # run through the different states
            for i in range(self.L):
                denom = sum(self.gamma_output[i,:-1])

                # new transition matrix A
                for j in range(self.L):
                    A_top[i,j] += newTransition(i,j,sub_x, alpha, beta)
                    A_bot[i,j] += denom

                # new emission matrix O
                for j in range(self.D):
                    O_top[i,j] += newEmission(i,j,sub_x, alpha, beta)
                    O_bot[i,j] += denom

        self.A = A_top/A_bot
        self.O = O_top/O_bot

        # normalize A and O
        self.A = self.A / np.sum(self.A,1)[:,None]
        self.O = self.O / np.sum(self.O,1)[:,None]

        pass

    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a string.
        '''
        emission = ''
        y = np.array(np.random.randint(0,self.L))
        for j in range(M):
            y = np.random.choice(self.L, 1, p = self.A[y,:])[0]
            x = np.random.choice(self.D, 1, p = self.O[y,:])[0]
            emission += str(x)

        return emission

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''
        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the output sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any output sequence, i.e. the
        # probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob

def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learing.

    Arguments:
        X:          A list of variable length emission sequences
        Y:          A corresponding list of variable length state sequences
                    Note that the elements in X line up with those in Y
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrices A and O.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)

    start = time.time()
    for i in range(1):
        print(i)
        HMM.unsupervised_learning(X)

    end = time.time()
    print('time:' ,end - start)

    return HMM
