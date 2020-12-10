from functools import total_ordering, reduce
import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import time


#necessario per salvare un indice grande con pickle
import sys
sys.setrecursionlimit(10000)

#garbage collector
import gc

import pickle
nltk.download('punkt')
nltk.download('stopwords')

#normalize e tokenize da sistemare con nltp
def normalize(text):
    no_punctuation = re.sub(r'[^\w^\s^#-]', '', text)
    downcase = no_punctuation.lower()
    return downcase


def stem(text):
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in text if word != "and" or word != "or" or word != "not"]
    return stemmed


def process(desc):
    stop_words = set(stopwords.words('english'))

    normalized = normalize(desc)
    tokenized = normalized.split()
    no_stop_words = [w for w in tokenized if not w in stop_words]
    processed = stem(no_stop_words)
    return processed


def process_query(query):
    stop_words = set(stopwords.words('english'))

    normalized = normalize(query)
    tokenized = normalized.split()
    no_stop_words = [w for w in tokenized if not w in stop_words and w != "and" or w != "or" or w != "not"]
    stemmed = stem(no_stop_words)
    return stemmed


class ImpossibleMergeError(Exception):
    pass

@total_ordering
class Term:
#a un termine è associata una posting list

    def __init__(self, term, docID, position):
        #quando ho un nuovo termine la posting list è formata solo dal primo docID
        self.term = term
        self.posting_list = PostingList.from_docID(docID, position)

    def merge(self, other):
        """Merge (destructively) this term and the corresponding posting list
        with another equal term and its corrsponding posting list.
        """
        if (self.term == other.term):
            self.posting_list.merge(other.posting_list)
        else:
            raise ImpossibleMergeError()

    def __eq__(self, other):
        return self.term == other.term

    def __gt__(self, other):
        return self.term > other.term

    def __repr__(self):
        return self.term + ": " + repr(self.posting_list)


#elementi di una posting list (doc id + posizioni del termine)
@total_ordering
class Posting:

    def __init__(self, docID, pos):
        self._docID = docID
        self._poslist = []  # Term frequency
        self._poslist.append(pos)

    def get_from_corpus(self, corpus):
        return corpus[self._docID]

    def __eq__(self, other):
        """Perform the comparison between this posting and another one.
        Since the ordering of the postings is only given by their docID,
        they are equal when their docIDs are equal.
        """
        return self._docID == other._docID

    def __gt__(self, other):
        """As in the case of __eq__, the ordering of postings is given
        by the ordering of their docIDs
        """
        return self._docID > other._docID

    def __repr__(self):
        return str(self._docID) + " [" + str(self._poslist) + "]"

    def __hash__(self):
        return hash(self._docID)


#lista degli (docID,lista delle posizioni) di un term
class PostingList:

    def __init__(self):
        self._postings = []

    @classmethod
    def from_docID(cls, docID, position):
        """ A posting list can be constructed starting from
        a single docID.
        """
        plist = cls()
        plist._postings = [Posting(docID, position)]
        return plist

    @classmethod
    def from_posting_list(cls, postingList):
        """ A posting list can also be constructed by using another
        """
        plist = cls()
        plist._postings = postingList
        return plist

    def merge(self, other):
        #per unire le posting list di un termine presente in più docID
        """Merge the other posting list to this one in a desctructive
        way, i.e., modifying the current posting list. This method assume
        that all the docIDs of the second list are higher than the ones
        in this list. It assumes the two posting lists to be ordered
        and non-empty. Under those assumptions duplicate docIDs are
        discard
        """
        i = 0
        last = self._postings[-1]
        while (i < len(other._postings) and last == other._postings[i]):
            last._poslist.append(other._postings[i]._poslist[0])
            i += 1
        self._postings += other._postings[i:]

    def intersection(self, other):
        #per la query
        """Returns a new posting list resulting from the intersection
        of this one and the one passed as argument.
        """
        intersection = []
        i = 0
        j = 0
        while (i < len(self._postings) and j < len(other._postings)):
            if (self._postings[i] == other._postings[j]):
                intersection.append(self._postings[i])
                i += 1
                j += 1
            elif (self._postings[i] < other._postings[j]):
                i += 1
            else:
                j += 1
        return PostingList.from_posting_list(intersection)

    def union(self, other):
        #per la query
        """Returns a new posting list resulting from the union of this
        one and the one passed as argument.
        """
        union = []
        i = 0
        j = 0
        while (i < len(self._postings) and j < len(other._postings)):
            if (self._postings[i] == other._postings[j]):
                union.append(self._postings[i])
                i += 1
                j += 1
            elif (self._postings[i] < other._postings[j]):
                union.append(self._postings[i])
                i += 1
            else:
                union.append(other._postings[j])
                j += 1
        for k in range(i, len(self._postings)):
            union.append(self._postings[k])
        for k in range(j, len(other._postings)):
            union.append(other._postings[k])
        return PostingList.from_posting_list(union)

    '''da sviluppare la NOT QUERY'''
    def not_query(self, other):
        '''per fare la not devo prendere la posting list del termine
        se è chiamata singola allora devo prendere una posting list con tutti i termini
        altrimenti basta togliere quelli che ci sono nell'altra'''

        #self = quella della not
        #other = altra posting list o posting list di tutti i documenti
        not_term = []
        i = 0
        j = 0
        while (i < len(self._postings) and j < len(other._postings)):
            if (self._postings[i] == other._postings[j]):
                #se sono uguali non lo aggiungo
                i += 1
                j += 1
            elif (self._postings[i] < other._postings[j]):
                not_term.append(self._postings[i])
                i += 1
            else:
                j += 1
        return PostingList.from_posting_list(not_term)

    def phrase(self, other):
        phrase = []
        i = 0
        j = 0
        while i < len(self._postings) and j < len(other._postings):
            x = 0
            y = 0
            if self._postings[i] == other._postings[j]:
                # sono nello stesso documento, devo controllare se sono vicini
                while x < len(self._postings[i]._poslist) and y < len(other._postings[j]._poslist):
                    if self._postings[i]._poslist[x] == other._postings[j]._poslist[y]-1:
                        phrase.append(other._postings[j])
                        break
                    x += 1
                    y += 1
                i += 1
                j += 1

            elif self._postings[i] < other._postings[j]:
                i += 1
            else:
                j += 1
        return PostingList.from_posting_list(phrase)

    def not_query_all_docs(self, number_of_docs):
        alldocs = []
        docID = 1
        while docID < number_of_docs:
            alldocs.append(Posting(docID, 1))
            docID += 1

        i = 0
        while i < len(self._postings):
            alldocs.remove(self._postings[i])
            i += 1

        return PostingList.from_posting_list(alldocs)

    def get_from_corpus(self, corpus):
        return list(map(lambda x: x.get_from_corpus(corpus), self._postings))

    def __len__(self):
        return len(self._postings)

    def __iter__(self):
        return iter(self._postings)

    def __repr__(self):
        return ", ".join(map(str, self._postings))



class DataDescription:
    #generare la coppia titolo-testo
    def __init__(self, title, description):
        self.title = title
        self.description = description

    def __repr__(self):
        return self.title  # + "\n" + self.description + "\n"


class InvertedIndex:

    def __init__(self):
        self._dictionary = []  # A collection of terms
        self._N = 0  # Number of documents in the collection

    @classmethod
    def from_corpus(cls, corpus):
        # Here we "cheat" by using python dictionaries
        intermediate_dict = {}
        for docID, document in enumerate(corpus):
            stop_words = set(stopwords.words('english'))

            tokens = process(document.description)

            for index, token in enumerate(tokens):
                #creo il nuovo termine
                token = token + "$"

                #devo ruotare la parola per fare la trailing wildcard
                for i in token:
                    term = Term(token, docID, index)

                    try:
                    #se è già presente nel dizionario faccio il merge delle posting list
                        intermediate_dict[token].merge(term)
                    except KeyError:
                    #altrimenti faccio un nuovo documento
                        intermediate_dict[token] = term
                    token = token[1:]+token[0]

            # To observe the progress of our indexing.
            if (docID % 1000 == 0 and docID != 0):
                print(str(docID), end='...')

                # enable break to limit indexing for testing
                #if(docID % 20000 == 0):
                #    break


        idx = cls()
        idx._N = len(corpus)
        #ritorno il dizionario con i termini ordinati in ordine alfabetico
        idx._dictionary = sorted(intermediate_dict.values())
        return idx

    #per rispondere a una singola query

    '''aggiungere modo per rispondere a più query'''
    def __getitem__(self, key):
        wildcards = []
        simp_query = []
        key = key + "$"
        counter = key.count('#')
        if counter ==1 :
            #vuol dire che è una wildcard

            #ruoto fino ad avere l'asterisco alla fine e poi tolgo l'asterisco e cerco
            while(not key.endswith("#")):
                key = key[1:] + key[0]

            key = key[:-1]
            for term in self._dictionary:
                if term.term.startswith(key):
                    wildcards.append(term.posting_list)
            #devo trasformare il vettore di wildcards in una sola posting list
            plist = reduce(lambda x, y: x.union(y), wildcards)
            return plist

        elif counter >1:
            #salvo la prima parte, tengo solo la ultima e faccio la query su questa
            wcards = key.rsplit('#', 1)

            '''manca il caso in cui la parte sconosciuta sia alla fine'''

            key1 = '#'+wcards[1]

            #ruoto fino ad avere l'asterisco alla fine e poi tolgo l'asterisco e cerco
            while(not key1.endswith("#")):
                key1 = key1[1:] + key1[0]

            key1 = key1[:-1]
            for term in self._dictionary:
                if term.term.startswith(key1):
                    simp_query.append(term)

            #tolgo il dollaro alla prima wildcard e sostituisco gli # del resto della parola con i simboli per regex
            key1 = key1[:-1]
            key2 = wcards[0].replace("#", ".+")

            #metto insieme le singole wildcard (prima metto quella già restituita così vado in ordine) e compilo la regex
            match = key1 + "\$" + key2 + ".+"
            reg = re.compile(match)

            terms_retrieve = []
            for term in simp_query:
                if re.match(reg, term.term):
                   terms_retrieve.append(term.posting_list)

            #ora ho tutti i termini che rispecchiano la wildcard "alla fine"
            plist = reduce(lambda x, y: x.union(y), terms_retrieve)
            return plist

        else:
            #è una parola completa
            for term in self._dictionary:
                if(term.term == key):
                    return term.posting_list
        raise KeyError

    def __repr__(self):
        return "A dictionary with " + str(len(self._dictionary)) + " terms"


class IRsystem:

    def __init__(self, corpus, index):
        self._corpus = corpus
        self._index = index

    @classmethod
    def from_corpus(cls, corpus):
        index = InvertedIndex.from_corpus(corpus)
        #print(index)
        return cls(corpus, index)

    def answer_query_full(self, query):
        norm_words = query
        i = 0
        while i < len(norm_words):

            # se non è un termine di connessione ottengo la posting list
            if norm_words[i] != "and" and norm_words[i] != "or" and norm_words[i] != "not":

                res = self._index[norm_words[i]]
                norm_words[i] = res
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
                if not norm_words[j]:
                    return
                del norm_words[j+1]
            j += 1

        j = 0
        # una volta risolte le phrase queries posso applicare gli operatori AND, OR, NOT
        # sovrascrivo il primo elemento dell'operazione con il risultato e elimino gli altri,
        # alla fine avrò un'unica posting list da cui andare a prendere i titoli dei film
        while len(norm_words) != 1:
            if not isinstance(norm_words[j], str) and norm_words[j+1] == "and" and not isinstance(norm_words[j+2], str):
                # caso AND, devo avere Posting List, "and", Posting List
                norm_words[j] = norm_words[j].intersection(norm_words[j+2])
                if not norm_words[j]:
                    return
                del norm_words[j + 1]
                del norm_words[j + 1]

            elif not isinstance(norm_words[j], str) and norm_words[j+1] == "or" and not isinstance(norm_words[j+2], str):
                norm_words[j] = norm_words[j].union(norm_words[j+2])
                if not norm_words[j]:
                    return
                del norm_words[j + 1]
                del norm_words[j + 1]

            elif not isinstance(norm_words[j], str) and norm_words[j+1] == "and" and norm_words[j+2] == "not" and not isinstance(norm_words[j+3], str):
                norm_words[j] = norm_words[j].not_query(norm_words[j+3])
                if not norm_words[j]:
                    return
                del norm_words[j + 1]
                del norm_words[j + 1]
                del norm_words[j + 1]

            elif norm_words[j] == "not" and not isinstance(norm_words[j+1], str):
                norm_words[j] = norm_words[j+1].not_query_all_docs(self._index._N)
                if not norm_words[j]:
                    return
                del norm_words[j + 1]

        return norm_words[0].get_from_corpus(self._corpus)


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

def query(ir, text):
    terms_query = process_query(text)

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
    file = open("data/dictionary.pickle", "wb")
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
            f = open('data/dictionary.pickle', 'rb')

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