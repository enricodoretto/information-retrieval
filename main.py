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
        self._number_of_docs = 0

    @classmethod
    def from_corpus(cls, corpus):
        trie = Trie()
        number_of_docs = 0

        for docID, document in enumerate(corpus):
            number_of_docs += 1
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
                    #faccio la query sulla wildcard finale e aggiungo alle posting list da ritornare solo se il termine matcha il pattern
                    multiplewcards = self._index._trie.getWildcardMW(key, pattern)
                    resmultiplewcards = reduce(lambda x, y: x.union(y), multiplewcards)
                    postings.append(resmultiplewcards)

            else:
                #caso di parola non wildcard
                res = self._index._trie.search(w)
                postings.append(res)
                #if word not found
                if(not res):
                    return

        #una volta ottenute le posting list dei term della query devo rispondere in base alle connections
        plist = []
        if(not connections and len(norm_words) > 1):
            #se non ho connessioni e ho più di due parole allora è una frase
            plist = reduce(lambda x, y: x.phrase(y), postings)

        elif(connections):
            for conn in connections:
                if(conn == "and"):
                    plist = reduce(lambda x, y: x.intersection(y), postings)
                elif(conn == "or"):
                    plist = reduce(lambda x, y: x.union(y), postings)
                else:
                    '''sviluppare la not'''
                    #plist = reduce(lambda x, y: x.not_query(y), postings)
                    plist = postings[0].not_query_all_docs(self._index._number_of_docs)

        else:
            #sono nel caso di una sola parola
            plist = reduce(lambda x, y: x.intersection(y), postings)

        return plist.get_from_corpus(self._corpus)


    def answer_query_full(self, query):

        norm_words = query
        postings = []

        i=0
        while(i<len(norm_words)):
            if (norm_words[i] != "and" and norm_words[i] != "or" and norm_words[i] != "not"):
                norm_words[i] = norm_words[i] + '$'
                counter = norm_words[i].count('#')
                if not counter == 0:
                    #vuol dire che è una wildcard
                    '''implementare multiple e errore se non trovata'''

                    if(counter == 1):
                        #caso di wildcard semplice, basta ruotare
                        while (not norm_words[i].endswith("#")):
                            norm_words[i] = norm_words[i][1:] + norm_words[i][0]
                        norm_words[i] = norm_words[i][:-1]

                        singlewcard = self._index._trie.getWildcard(norm_words[i])
                        ressinglewcard = reduce(lambda x, y: x.union(y), singlewcard)
                        #postings.append(ressinglewcard)
                        norm_words[i] = ressinglewcard

                    elif(counter > 1):
                        '''multiple wildcard'''
                        #salvo la prima parte, tengo solo la ultima e faccio la query su questa
                        cards = norm_words[i].rsplit('#', 1)
                        key1 = '#' + cards[1]
                        while(not key1.endswith("#")):
                            key1 = key1[1:] + key1[0]

                        key = key1[:-1]

                        # tolgo il dollaro alla prima wildcard e sostituisco gli # del resto della parola con i simboli per regex
                        key1 = key1[:-1]
                        key1 = key1[:-1]
                        key2 = cards[0].replace("#", ".+")
                        pattern = key1 + "\$" + key2 + ".+"
                        #faccio la query sulla wildcard finale e aggiungo alle posting list da ritornare solo se il termine matcha il pattern
                        multiplewcards = self._index._trie.getWildcardMW(key, pattern)
                        resmultiplewcards = reduce(lambda x, y: x.union(y), multiplewcards)
                        norm_words[i] = resmultiplewcards


                else:
                    #caso di parola non wildcard
                    res = self._index._trie.search(norm_words[i])
                    norm_words[i] = res
                    #if word not found
                    if(not res):
                        return
                i += 1

            else:
                i += 1

        #una volta ottenute le posting list dei term della query devo rispondere in base alle connections
        plist = []
        j = 0
        #risolvo le phrase queries
        while (j < len(norm_words)-1):
            if not isinstance(norm_words[j], str) and not isinstance(norm_words[j+1],str):
                norm_words[j] = norm_words[j].phrase(norm_words[j+1])
                del norm_words[j+1]
            j += 1

        j=0
        while (len(norm_words) != 1):
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

        return norm_words[0].get_from_corpus(self._corpus)

def query(ir, text):
    normalized = normalize(text)
    tokenized = normalized.split()
    connections = []
    terms = []

    for word in tokenized:
        if (word == "and" or word == "or" or word == "not"):
            connections.append(word)
        else:
            terms.append(word + "$")

    terms = stem(terms)

    for word in tokenized:
        if (word != "and" or word != "or" or word != "not"):
            word + "$"

    termsq = stem(tokenized)

    #answer = ir.answer_query(terms, connections)
    answer = ir.answer_query_full(termsq)

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

