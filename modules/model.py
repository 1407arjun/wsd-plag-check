import warnings
from abc import ABCMeta, abstractmethod
from nltk.lm.counter import NgramCounter
from nltk.lm.vocabulary import Vocabulary


class LanguageModel(metaclass=ABCMeta):
    def __init__(self, order, vocabulary=None, counter=None):
        self.order = order
        if vocabulary and not isinstance(vocabulary, Vocabulary):
            warnings.warn(
                f"The `vocabulary` argument passed to {self.__class__.__name__!r} "
                "must be an instance of `nltk.lm.Vocabulary`.",
                stacklevel=3,
            )
        self.vocab = Vocabulary() if vocabulary is None else vocabulary
        self.counts = NgramCounter() if counter is None else counter

    def fit(self, text, vocabulary_text=None):
        if not self.vocab:
            if vocabulary_text is None:
                raise ValueError(
                    "Cannot fit without a vocabulary or text to create it from."
                )
            self.vocab.update(vocabulary_text)
        self.counts.update(self.vocab.lookup(sent) for sent in text)

    def score(self, word, context=None):
        return self.unmasked_score(
            self.vocab.lookup(word), self.vocab.lookup(
                context) if context else None
        )

    @abstractmethod
    def unmasked_score(self, word, context=None):
        raise NotImplementedError()

    def context_counts(self, context):
        return (
            self.counts[len(context) +
                        1][context] if context else self.counts.unigrams
        )


class MLE(LanguageModel):
    def unmasked_score(self, word, context=None):
        return self.context_counts(context).freq(word)
