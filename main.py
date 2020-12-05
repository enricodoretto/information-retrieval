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
import sys
import gc


sys.setrecursionlimit(10000)  # necessario per salvare un indice grande con pickle
nltk.download('punkt')
nltk.download('stopwords')


# lettura dei dati e costruzione del corpus
def read_data_descriptions():
    filename = 'data/plot_summaries.txt'
    data_names_file = 'data/movie.metadata.tsv'
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
        self._number_of_docs = 0

    # costruzione dell'indice partendo dal corpus
    @classmethod
    def from_corpus(cls, corpus):
        trie = Trie()
        number_of_docs = 0

        for docID, document in enumerate(corpus):
            number_of_docs += 1
            tokens = process(document.description)
            for index, token in enumerate(tokens):
                # creo il termine da passare a ogni rotazione
                term = Term(token, docID, index)
                token = token + "$"
                # devo ruotare la parola per fare la trailing wildcard
                for i in token:
                    trie.insert(token, term)
                    token = token[1:] + token[0]

            if docID % 1000 == 0 and docID != 0:
                print(str(docID), end='...')
                break
                # per limitare l'index a 20000 o 1000 elementi
                #if(docID % 20000 == 0):
                #    break

        idx = cls()
        idx._number_of_docs = number_of_docs
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

    def answer_query_full(self, query):
        norm_words = query
        i = 0
        while i < len(norm_words):

            # se non è un termine di connessione ottengo la posting list
            if norm_words[i] != "and" and norm_words[i] != "or" and norm_words[i] != "not":
                norm_words[i] = norm_words[i] + '$'

                #controllo se è una wildcard
                counter = norm_words[i].count('#')
                if not counter == 0:
                    if counter == 1:
                        # caso di wildcard semplice, basta ruotare e effettuare la ricerca
                        while not norm_words[i].endswith("#"):
                            norm_words[i] = norm_words[i][1:] + norm_words[i][0]
                        norm_words[i] = norm_words[i][:-1]

                        singlewcard = self._index._trie.get_wildcard(norm_words[i])
                        if not singlewcard:
                            return
                        # il risultato sarà una posting list formata dall'unione di tutte le posting list dei termini corrispondenti
                        ressinglewcard = reduce(lambda x, y: x.union(y), singlewcard)
                        norm_words[i] = ressinglewcard

                    elif counter > 1:
                        # multiple wildcard
                        # salvo la prima parte, la query andrà fatta solo sull'ultima parte della wildcard
                        cards = norm_words[i].rsplit('#', 1)
                        key1 = '#' + cards[1]
                        while not key1.endswith("#"):
                            key1 = key1[1:] + key1[0]
                        key = key1[:-1]

                        # tolgo il dollaro alla prima wildcard e sostituisco gli # del resto della parola con i simboli per regex
                        key1 = key1[:-1]
                        key1 = key1[:-1]
                        key2 = cards[0].replace("#", ".+")
                        pattern = key1 + "\$" + key2 + ".+"
                        # faccio la query sulla wildcard finale e aggiungo alle posting list da ritornare solo se il termine matcha il pattern
                        multiplewcards = self._index._trie.get_multiple_wildcard(key, pattern)
                        if not multiplewcards:
                            return
                        resmultiplewcards = reduce(lambda x, y: x.union(y), multiplewcards)
                        norm_words[i] = resmultiplewcards

                else:
                    # caso di parola normale
                    res = self._index._trie.search(norm_words[i])
                    norm_words[i] = res
                    # if word not found
                    if not res:
                        return
                i += 1

            else:
                i += 1

        j = 0
        # dopo aver ottenuto le posting list risolvo prima di tutto le phrase queries
        while j < len(norm_words)-1:
            if not isinstance(norm_words[j], str) and not isinstance(norm_words[j+1], str):
                norm_words[j] = norm_words[j].phrase(norm_words[j+1])
                del norm_words[j+1]
            j += 1

        j = 0
        #una volta risolte le phrase queries posso applicare gli operatori
        while len(norm_words) != 1:
            if not isinstance(norm_words[j], str) and norm_words[j+1] == "and" and not isinstance(norm_words[j+2], str):
                norm_words[j] = norm_words[j].intersection(norm_words[j+2])
                del norm_words[j + 1]
                del norm_words[j + 1]

            elif not isinstance(norm_words[j], str) and norm_words[j+1] == "or" and not isinstance(norm_words[j+2], str):
                norm_words[j] = norm_words[j].union(norm_words[j+2])
                del norm_words[j + 1]
                del norm_words[j + 1]

            elif not isinstance(norm_words[j], str) and norm_words[j+1] == "and" and norm_words[j+2] == "not" and not isinstance(norm_words[j+3], str):
                norm_words[j] = norm_words[j].not_query(norm_words[j+3])
                del norm_words[j + 1]
                del norm_words[j + 1]
                del norm_words[j + 1]

            elif norm_words[j] == "not" and not isinstance(norm_words[j+1], str):
                norm_words[j] = norm_words[j+1].not_query_all_docs(self._index._number_of_docs)
                del norm_words[j + 1]

        return norm_words[0].get_from_corpus(self._corpus)


def query(ir, text):
    #effettuo le operazioni di normalizzazione, tokenizzazione e stemming della query
    normalized = normalize(text)
    tokenized = normalized.split()
    terms_query = stem(tokenized)

    answer = ir.answer_query_full(terms_query)

    if not answer:
        print("Nessuna corrispondenza trovata")
    else:
        for doc in answer:
            print(doc)
        print(len(answer))


# lettura file, creazione e salvataggio indice
def initialization():
    # disabilitazione garbage collector per migliorare le prestazioni
    gc.disable()

    tic = time.perf_counter()
    corpus = read_data_descriptions()
    ir = IRsystem.from_corpus(corpus)
    tac = time.perf_counter()
    print(f"Index builded in {tac - tic:0.4f} seconds")
    #file = open('data/trie_index_20000.pickle', 'wb')
    file = open("data/trie_index_1000.pickle", "wb")
    #file = open("data/trie_index.pickle", "wb")
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
            f = open('data/trie_index_1000.pickle', 'rb')
            #f = open('data/trie_index.pickle', 'rb')

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
