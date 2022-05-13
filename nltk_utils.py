import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()       #creating object from PorterStemmer() class





def tokenize(sentence):
    """
    input:
        sentence = 'How long does shipping take?'
    output:
        tokenized = ['How', 'long', 'does', 'shipping', 'take', '?']
    """
    return nltk.word_tokenize(sentence)







def stem(words):
    """
    input:
        words = ['Organize', 'organizes', 'organizing']
    output:
        stemmed = ['organ', 'organ', 'organ']
    """
    stemmed_sentence = [stemmer.stem(word.lower()) for word in words]
    return stemmed_sentence







def bag_of_words(tokenized_sentence, all_words):
    """
    input:
        tokenized_sentence = ['hello', 'how', 'are', 'you']
        all_words = ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool', 'how']
    output:
        bag = [0. 1. 0. 1. 0. 0. 0. 1.]
    """
    tokenized_sentence = stem(tokenized_sentence)

    bag = np.zeros(len(all_words), dtype='f')
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag





# only for expreiment
if __name__ == "__main__":
    a = "How long does shipping take?"
    print(a)
    a = tokenize(a)
    print(a)


    words = ["Organize", "organizes", "organizing"]
    print(words)
    words = stem(words)
    print(words)


    sentence = ['hello', 'how', 'are', 'you']
    words = ['hi', 'hello', 'I', 'you', 'bye', 'thank', 'cool', 'how']
    bag = bag_of_words(sentence, words)
    print(bag)
