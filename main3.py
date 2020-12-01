import pickle
from functools import total_ordering, reduce
import csv
import nltk
import time
from term_operations4 import DataDescription
from term_operations4 import tokenize
from trie3 import TrieNode
from trie3 import add
from trie3 import find_item

nltk.download('punkt')
nltk.download('stopwords')



def read_data_descriptions():
    filename = '../../Google Drive/Università/Corsi/Materiale corsi quinto anno/Information retrieval and data visualization/Lecture 3/MovieSummaries/plot_summaries.txt'
    data_names_file = '../../Google Drive/Università/Corsi/Materiale corsi quinto anno/Information retrieval and data visualization/Lecture 3/MovieSummaries/movie.metadata.tsv'
    with open(data_names_file, 'r') as csv_file:
        data_names = csv.reader(csv_file, delimiter='\t')
        names_table = {}
        for name in data_names:
            names_table[name[0]] = name[2]
    with open(filename, 'r') as csv_file:
        descriptions = csv.reader(csv_file, delimiter='\t')
        corpus = []
        for desc in descriptions:
            try:
                #coppia titolo-testo
                corpelement = DataDescription(names_table[desc[0]], desc[1])
                corpus.append(corpelement)
            except KeyError:
                # We ignore the descriptions for which we cannot find a title
                pass
        return corpus

class InvertedIndex:
    def __init__(self):
        self._root = TrieNode  # A trie of terms
        self._N = 0  # Number of documents in the collection

    @classmethod
    def from_corpus(cls, corpus):
        root1 = TrieNode('*')

        # Here we "cheat" by using python dictionaries
        for docID, document in enumerate(corpus):
            tokens = tokenize(document)
            for index, token in enumerate(tokens):
                # creo il nuovo termine
                token = token + "$"

                # devo ruotare la parola per fare la trailing wildcard
                for i in token:
                    # se è già presente nel dizionario faccio il merge delle posting list
                    add(root1, token, docID, index)

                    token = token[1:] + token[0]

                # To observe the progress of our indexing.
            if (docID % 1000 == 0 and docID != 0):
                print(str(docID), end='...')
                # enable break to limit indexing for testing
                break

        idx = cls()
        idx._root = root1
        idx._N = len(corpus)
        return idx


class IRsystem:
    def __init__(self, corpus, index):
        self._corpus = corpus
        self._index = index

    @classmethod
    def from_corpus(cls, corpus):
        index = InvertedIndex.from_corpus(corpus)
        return cls(corpus, index)

    def answer_query(self, words, connections):
        '''da riabilitare'''
        #norm_words = map(normalize, words)

        norm_words = words
        postings = []

        for w in norm_words:
            res = find_item(self._index._root, w)
            #if word not found
            if(not res):
                break
            postings.append(res)

        #having all the postings now I have to compute and-or-not
        plist = []
        if(not connections and len(norm_words) > 1):
            #se non ho connessioni e ho più di due parole è una frase
            plist = reduce(lambda x, y: x.phrase(y), postings)

        elif (len(norm_words) == 1):
            #sono nel caso di una sola parola
            plist = reduce(lambda x, y: x.intersection(y), postings)

        else:
            for conn in connections:
                if(conn == "and"):
                    plist = reduce(lambda x, y: x.intersection(y), postings)
                elif(conn == "or"):
                    plist = reduce(lambda x, y: x.union(y), postings)
                else:
                    pass
                    '''sviluppare la not'''
                    plist = reduce(lambda x, y: x.not_query(y), postings)

        return plist.get_from_corpus(self._corpus)


def query(ir, text):
    words = text.split()
    connections = []
    terms = []

    #Divide terms to retrieve and connections such and, or, not
    for word in words:
        if (word == "and" or word == "or" or word == "not"):
            connections.append(word)
        else:
            terms.append(word + "$")

    answer = ir.answer_query(terms,connections)
    print(len(answer))
    for doc in answer:
        print(doc)

#lettura file, creazione e salvataggio indice
def initialization():
    tic = time.perf_counter()
    corpus = read_data_descriptions()
    ir = IRsystem.from_corpus(corpus)
    with open("data/trie.pickle", "wb") as file_:
        pickle.dump(ir, file_, -1)
    toc = time.perf_counter()
    print(f"Index created in {toc - tic:0.4f} seconds")

def operate():
    print("Retrieving index...")
    ir = pickle.load(open("data/trie.pickle", "rb", -1))
    print("Index retrieved!")
    tic = time.perf_counter()
    query(ir, "poovalli induchoodan")
    toc = time.perf_counter()
    print(f"Query performed in {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    #initialization()
    operate()

