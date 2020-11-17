from functools import total_ordering, reduce
import csv
import re
import nltk
from nltk.corpus import stopwords

import pickle
nltk.download('punkt')
nltk.download('stopwords')

#normalize e tokenize da sistemare con nltp
def normalize(text):
    """ A simple funzion to normalize a text.
    It removes everything that is not a word, a space or an hyphen
    and downcases all the text.
    """
    no_punctuation = re.sub(r'[^\w^\s^-]', '', text)
    downcase = no_punctuation.lower()
    return downcase


def tokenize(movie):
    """ From a movie description returns a posting list of all
    tokens present in the description.
    """
    stop_words = set(stopwords.words('english'))
    text = normalize(movie.description)
    tokenized = nltk.word_tokenize(text)
    filtered_sentence = [w for w in tokenized if not w in stop_words]

    #return list(text.split())
    return filtered_sentence

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


    def phrase(self,other):
        phrase = []
        i=0
        j=0
        while (i < len(self._postings) and j < len(other._postings)):
            x=0
            y=0
            if (self._postings[i] == other._postings[j]):
                #sono nello stesso documento, devo controllare se sono vicini
                while (x < len(self._postings[i]._poslist) and y < len(other._postings[j]._poslist)):
                    if(self._postings[i]._poslist[x] == other._postings[j]._poslist[y]-1):
                        phrase.append(other._postings[y])
                    x += 1
                    y += 1
                i += 1
                j += 1

            elif (self._postings[i] < other._postings[j]):
                i += 1
            else:
                j += 1
        return PostingList.from_posting_list(phrase)


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
            tokens = tokenize(document)
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

                #enable break to limit indexing for testing
                break


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

            key1 = '#'+wcards[1]

            #ruoto fino ad avere l'asterisco alla fine e poi tolgo l'asterisco e cerco
            while(not key1.endswith("#")):
                key1 = key1[1:] + key1[0]

            key1 = key1[:-1]
            for term in self._dictionary:
                if term.term.startswith(key1):
                    simp_query.append(term)
            #devo trasformare il vettore di termini in uno solo
            #simp_query_list = reduce(lambda x, y: x.union(y), simp_query)

            '''DA SISTEMARE'''
            terms_retrieve = []
            reg = re.compile(".+a.t")

            for term in simp_query:
                if re.match(reg, term.term):
                   terms_retrieve.append(term.term)
                   #terms_retrieve.append(term.posting_list)

            #ora ho tutti i termini che rispecchiano la wildcard "alla fine"

            pass
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

    def answer_query_sc(self, words, connections):
        #da riabilitare
        #norm_words = map(normalize, words)
        norm_words = words
        postings = []

        for w in norm_words:
            try:
                res = self._index[w]
            except KeyError:
                #implementare il not found
                print("{} not found. Did you mean {}?")
                pass
            postings.append(res)
        #having all the postings now I have to compute and-or-not
        plist = []
        if(not connections):
            #se non viene specificata una connection word viene ritornata l'and
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
        return plist.get_from_corpus(self._corpus)

    def answer_phrase_query(self, words):
        #da riabilitare
        #norm_words = map(normalize, words)
        norm_words = words
        postings = []
        for w in norm_words:
            try:
                res = self._index[w]
            except KeyError:
                #implementare il not found
                print("{} not found. Did you mean {}?")
                pass
            postings.append(res)
        #having all the postings now I have to check if they are near
        plist = reduce(lambda x, y: x.phrase(y), postings)
        return plist.get_from_corpus(self._corpus)


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
    words = text.split()
    connections = []
    terms = []

    #Divide terms to retrieve and connections such and, or, not
    for word in words:
        if (word == "and" or word == "or" or word == "not"):
            connections.append(word)
        else:
            terms.append(word)

    answer = ir.answer_phrase_query(terms)
    #answer = ir.answer_query_sc(terms,connections)
    print(len(answer))
    for doc in answer:
        print(doc)

#lettura file, creazione e salvataggio indice
def initialization():
    corpus = read_data_descriptions()
    ir = IRsystem.from_corpus(corpus)
    with open("myindex.pickle", "wb") as file_:
        pickle.dump(ir, file_, -1)

#caricamento indice e query
def operate():
    print("Retrieving index...")
    ir = pickle.load(open("myindex.pickle", "rb", -1))
    print("Index retrieved!")
    query(ir, "hard-working taxi driver")
#initialization()
operate()