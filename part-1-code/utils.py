import random
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

def example_transform(example):
    example["text"] = example["text"].lower()
    return example

def custom_transform(example):
    original_text = example["text"]
    tokens = word_tokenize(original_text)
    
    # Synonym replacement (30% probability per word)
    for i, token in enumerate(tokens):
        if token.isalpha() and len(token) > 3 and random.random() < 0.50:
            synsets = wordnet.synsets(token.lower())
            candidates = []
            for syn in synsets[:2]:
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if ' ' not in synonym and synonym.isalpha() and synonym.lower() != token.lower():
                        candidates.append(synonym)
            if candidates:
                tokens[i] = random.choice(candidates)
    
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(tokens)
    return example