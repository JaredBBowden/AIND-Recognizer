import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import pdb

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Variables to hold update scores
        best_bic = math.inf
        best_model = GaussianHMM()

        # Iterate across a range of model states
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1):

            try:
                # Fit a model based on state
                model = GaussianHMM(n_components = num_hidden_states, n_iter = 100)
                model.fit(self.X, self.lengths)

                # Values needed to for BIC
                # From the slides: http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
                # BIC = −2 log L + p log N,

                # Get WITHIN sample logL
                logL = model.score(self.X, self.lengths)

                # Compute the number of parameters
                num_parameters = num_hidden_states * num_hidden_states + 2 * num_hidden_states * len(self.X[0]) - 1

                # Compute overall BIC formula
                current_bic = (-2) * logL + num_parameters * math.log(len(self.X))

                # Control flow to update BIC score
                if current_bic <= best_bic:
                    best_model, best_bic = model, current_bic

                else:
                    continue

            except:
                continue

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # I'm going to use the same general skeleton I had been using for
        # the previous methods
        # Variables to hold update scores
        best_dic = -math.inf
        best_model = GaussianHMM()

        # Iterate across a range of model states
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1):

            try:

                # Fit a model based on state
                model = GaussianHMM(n_components = num_hidden_states, n_iter = 100)
                model.fit(self.X, self.lengths) # What are we fitting on?
                #model = self.base_model(num_hidden_states)

                # Compute elements we need. Full equation:
                # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                sigma_scores = 0
                M = 0

                #pdb.set_trace()

                # FIXME
                # Get WITHIN sample logL
                logL = model.score(self.X, self.lengths)

                # Compute sum(logL) for all words that are not the X[i]
                for word in self.hwords:

                    if word != self.this_word:

                        # Pull values to score this word
                        temp_X, temp_length = self.hwords[word]

                        # Score
                        sigma_scores += model.score(temp_X, temp_length)
                        M += 1

                # Now we compute DIC
                current_dic = logL - (1.0 / (float(M)) * sigma_scores)

                # Control flow to update DIC score
                if current_dic >= best_dic:
                    #print("We're updating scores!")
                    best_model, best_dic = model, current_dic

                else:
                    #print("We're NOT updating scores...")
                    continue

            except:
                #print("We missed the try block")
                continue

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        #pdb.set_trace()

        # Variable to hold the best scores-and-model across CV iterations
        best_score = -math.inf

        # Initialize scikit hmm object with default parameters
        best_model = GaussianHMM()

        # Initialize values for CV split object.
        # FIXME This could be achieved more eloquently: rough adjustments to
        # allow code to account for scenarios encountered within Recognizer data.
        if len(self.sequences) <= 2:
            splits = 2
        else:
            splits = 3

        split_method = KFold(n_splits = splits)

        # Iterate through states: for each state cross-validate n times.
        # Language and structure based on the CV snippet from the notebook
        # The execution snippet provides man and min components.
        for num_hidden_states in range(self.min_n_components, self.max_n_components + 1):
            try:

                # Return index values for fold splits
                # FIXME should there be a random parameter here?
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    try:
                        #pdf.set_trace()

                        # To USE the index that we get from our folds we need to use the
                        # provided function.

                        # For training data
                        X_train, X_train_lengths = combine_sequences(cv_train_idx, self.sequences)

                        # For test data
                        X_test, X_test_lengths = combine_sequences(cv_test_idx, self.sequences)

                        # Fit the model on the fold data (training) and current number
                        # of states.
                        # Note that the number of iterations use here is a carry over
                        # from the notebook.
                        model = GaussianHMM(n_components = num_hidden_states, n_iter = 1000)
                        model.fit(X_train, X_train_lengths)

                        # Return score on the test data
                        logL = model.score(X_test, X_test_lengths)

                        # Control flow to test for high-scores
                        if logL >= best_score:

                            best_model, best_score = model, logL

                    except:
                        continue

            except:
                continue

        return best_model
