import warnings
from asl_data import SinglesData
import math

import pdb

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Initialize variables to hold values that will be returned
    probabilities = []
    guess = []

    # Iterate over the test set
    for X_test, X_length in list(test_set.get_all_Xlengths().values()):

        # Hold values through iterations
        word_score = -math.inf
        word_guess = None
        probability_dictionary = {}

        # Iterate over elements of trained models
        for word, model in models.items():

            # The word associated with the highest score is the best guess
            try:
                temp_score = model.score(X_test, X_length)

                # Save all score results (by word) to a dictionary
                probability_dictionary[word] = temp_score

                if temp_score >= word_score:
                    word_score, word_guess = temp_score, word

            except:
                # Account for the condition where we are unable to generate
                # a score
                probability_dictionary[word] = -math.inf

        probabilities.append(probability_dictionary)
        guess.append(word_guess)

    return probabilities, guess
