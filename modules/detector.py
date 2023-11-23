from .utils import pad_sequence, generate_ngrams
from .tokenizer import TreebankWordTokenizer
from .lesk import load_synsets

from .model import MLE
import numpy as np

tokenizer = TreebankWordTokenizer()


def detect(train_text, test_text, n):
    # pad the text and tokenize
    training_data = pad_sequence(tokenizer.tokenize(
        train_text.lower()), n, pad_left=True)
    testing_data = pad_sequence(tokenizer.tokenize(
        test_text.lower(), False), n, pad_left=True)

    # generate ngrams
    ngrams = list(generate_ngrams(training_data, max_length=n))
    print("Number of ngrams in train text:", len(ngrams))

    # build ngram language models
    model = MLE(n)

    synset_data = []
    for i, item in enumerate(training_data[n-1:]):
        synsets = load_synsets(training_data[i:i+n-1], item)
        synset_data += ([item] + synsets if synsets else [item])

    model.fit([ngrams], vocabulary_text=synset_data)
    print(model.vocab)

    # assign scores
    scores = []
    for i, item in enumerate(testing_data[n-1:]):
        synsets = load_synsets(testing_data[i:i+n-1], item)
        s = max([model.score(it, testing_data[i:i+n-1])
                for it in ([item] + synsets if synsets else [item])])
        scores.append(s)

    return np.array(scores), testing_data
