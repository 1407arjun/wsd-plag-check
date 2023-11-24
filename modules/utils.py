from itertools import islice


def plag_percentage(scores):
    return sum([1 if s > 0.2 else 0 for s in scores])*100//len(scores)


def pad_sequence(sequence, n, pad_left=False, pad_right=False):
    if pad_left:
        sequence = ['<s>']*n + sequence  # <s>
    if pad_right:
        sequence = sequence + ['</s>']*n  # </s>
    return sequence


def generate_ngrams(sequence, min_length=1, max_length=-1):
    sequence = iter(sequence)
    # Get max_len for padding.
    if max_length == -1:
        try:
            max_length = len(sequence)
        except TypeError:
            sequence = list(sequence)
            max_length = len(sequence)

    # Pad if indicated using max_length
    sequence = pad_sequence(sequence, max_length)

    # Sliding window to store ngrams
    history = list(islice(sequence, max_length))

    # Yield ngrams from sequence
    while history:
        for ngram_len in range(min_length, len(history) + 1):
            yield tuple(history[:ngram_len])

        # Append element to history if sequence has more items
        try:
            history.append(next(sequence))
        except StopIteration:
            pass

        del history[0]
