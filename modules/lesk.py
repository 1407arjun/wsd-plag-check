from nltk.corpus import wordnet


def load_synsets(context, word):
    synsets = []
    if (word):
        # Lesk implementation
        context = set(context)
        if synsets is None:
            synsets = wordnet.synsets(word)

        if not synsets:
            return None

        _, wsd = max((len(context.intersection(ss.definition().split())), ss)
                     for ss in synsets)

        if (wsd):
            # print(wsd.name(), wsd.definition())
            synsets += [syn.name() for syn in wsd.lemmas()]

    return synsets
