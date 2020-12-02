import _pickle as pickle
from functools import reduce
import csv
import nltk
import time
from term_operations import Term
from term_operations import DataDescription
from term_operations import process
from term_operations import normalize
from term_operations import stem

from trie import Trie

nltk.download('punkt')
nltk.download('stopwords')

#necessario per salvare un indice grande con pickle
import sys
sys.setrecursionlimit(10000)

#garbage collector
import gc


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
                corpelement = DataDescription(names_table[desc[0]], desc[1])
                corpus.append(corpelement)
            except KeyError:
                pass
        return corpus

class InvertedIndex:
    def __init__(self):
        self._trie = Trie

    @classmethod
    def from_corpus(cls, corpus):
        trie = Trie()

        for docID, document in enumerate(corpus):
            tokens = process(document.description)
            for index, token in enumerate(tokens):
                term = Term(token, docID, index)
                token = token + "$"
                # devo ruotare la parola per fare la trailing wildcard
                for i in token:
                    trie.insert(token, term)

                    token = token[1:] + token[0]

            if (docID % 1000 == 0 and docID != 0):
                print(str(docID), end='...')
                #break
                # per limitare l'index a 20000 o 1000 elementi
                if(docID % 20000 == 0):
                    break

        idx = cls()
        idx._trie = trie
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

        norm_words = words
        postings = []

        for w in norm_words:
            counter = w.count('#')
            if not counter == 0:
                #vuol dire che è una wildcard
                '''implementare multiple e errore se non trovata'''

                if(counter == 1):
                    #caso di wildcard semplice, basta ruotare
                    while (not w.endswith("#")):
                        w = w[1:] + w[0]
                    w = w[:-1]

                    singlewcard = self._index._trie.getWildcard(w)
                    ressinglewcard = reduce(lambda x, y: x.union(y), singlewcard)
                    postings.append(ressinglewcard)

                elif(counter > 1):
                    '''multiple wildcard'''
                    #salvo la prima parte, tengo solo la ultima e faccio la query su questa
                    cards = w.rsplit('#', 1)
                    key1 = '#' + cards[1]
                    while(not key1.endswith("#")):
                        key1 = key1[1:] + key1[0]

                    key = key1[:-1]

                    # tolgo il dollaro alla prima wildcard e sostituisco gli # del resto della parola con i simboli per regex
                    key1 = key1[:-1]
                    key1 = key1[:-1]
                    key2 = cards[0].replace("#", ".+")
                    pattern = key1 + "\$" + key2 + ".+"
                    multiplewcards = self._index._trie.getWildcardMW(key, pattern)
                    resmultiplewcards = reduce(lambda x, y: x.union(y), multiplewcards)
                    postings.append(resmultiplewcards)

            else:
                res = self._index._trie.search(w)
                postings.append(res)
                #if word not found
                if(not res):
                    break
            #postings.append(res)

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
    normalized = normalize(text)
    tokenized = normalized.split()
    connections = []
    terms = []

    #Divide terms to retrieve and connections such and, or, not
    for word in tokenized:
        if (word == "and" or word == "or" or word == "not"):
            connections.append(word)
        else:
            terms.append(word + "$")

    terms = stem(terms)

    answer = ir.answer_query(terms, connections)
    for doc in answer:
        print(doc)
    print(len(answer))


#lettura file, creazione e salvataggio indice
def initialization():
    #disabilitazione garbage collector per migliorare le prestazioni
    gc.disable()

    tic = time.perf_counter()
    corpus = read_data_descriptions()
    ir = IRsystem.from_corpus(corpus)
    tac = time.perf_counter()
    print(f"Index builded in {tac - tic:0.4f} seconds")

    #file = open('data/trie_index_20000.pickle', 'wb')
    #file = open("data/trie_index_1000.pickle", "wb")
    file = open("data/trie_index.pickle", "wb")
    pickle.dump(ir, file, protocol=-1)
    file.close()
    toc = time.perf_counter()
    print(f"Index saved in {toc - tac:0.4f} seconds")

    gc.enable()

def operate():
    ir = IRsystem
    while 1:
        op = input("I per caricare l'indice, q per query: ")
        if op == "i":

            gc.disable()
            print("Retrieving index...")
            tic = time.perf_counter()
            #f = open('data/trie_index_20000.pickle', 'rb')
            #f = open('data/trie_index_1000.pickle', 'rb')
            f = open('data/trie_index.pickle', 'rb')

            ir = pickle.load(f)
            f.close()
            tac = time.perf_counter()
            print(f"Index retrieved in {tac - tic:0.4f} seconds")
            gc.enable()

        elif op == "q":
            que = input("Scrivi la query: ")
            tac = time.perf_counter()
            query(ir, que)
            toc = time.perf_counter()
            print(f"Query performed in {toc - tac:0.4f} seconds")


if __name__ == "__main__":
    op = input("I per inizializzare, q per query: ")
    if op == "i":
        initialization()
    elif op == "q":
        operate()

