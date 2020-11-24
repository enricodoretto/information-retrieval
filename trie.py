import pickle
from functools import total_ordering, reduce
import csv
from typing import Tuple
import re
import nltk
from nltk.corpus import stopwords
import time


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

    '''da sviluppare la NOT QUERY PER TUTTO IL DIZIONARIO'''
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
                        phrase.append(other._postings[j])
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

class DataDescription:
    #generare la coppia titolo-testo
    def __init__(self, title, description):
        self.title = title
        self.description = description

    def __repr__(self):
        return self.title  # + "\n" + self.description + "\n"








class TrieNode(object):
    """
    Our trie node implementation. Very basic. but does the job
    """

    def __init__(self, char: str):
        self.char = char
        self.children = []
        # Is it the last character of the word.`
        self.word_finished = False

        self.posting_list = PostingList
        # How many times this character appeared in the addition process
        self.counter = 1


def add(root, word: str, docID, position):
    """
    Adding a word in the trie structure
    """
    node = root
    for char in word:
        found_in_child = False
        # Search for the character in the children of the present `node`
        for child in node.children:
            if child.char == char:
                # We found it, increase the counter by 1 to keep track that another
                # word has it as well
                child.counter += 1
                # And point the node to the child that contains this char
                node = child
                found_in_child = True
                break
        # We did not find it so add a new chlid
        if not found_in_child:
            new_node = TrieNode(char)
            node.children.append(new_node)
            # And then point node to the new child
            node = new_node
    # Everything finished. Mark it as the end of a word.
    node.word_finished = True
    try:
        node.posting_list.merge(PostingList.from_docID(docID, position))
    except TypeError:
        node.posting_list = PostingList.from_docID(docID, position)


def find_prefix(root, prefix: str):
    """
    Check and return
      1. If the prefix exsists in any of the words we added so far
      2. If yes then how may words actually have the prefix
    """
    node = root
    # If the root node has no children, then return False.
    # Because it means we are trying to search in an empty trie
    if not root.children:
        return False, 0
    for char in prefix:
        char_not_found = True
        # Search through all the children of the present `node`
        for child in node.children:
            if child.char == char:
                # We found the char existing in the child.
                char_not_found = False
                # Assign node as the child containing the char and break
                node = child
                break
        # Return False anyway when we did not find a char.
        if char_not_found:
            return False, 0
    # Well, we are here means we have found the prefix. Return true to indicate that
    # And also the counter of the last node. This indicates how many words have this
    # prefix
    if node.word_finished:
        return node.posting_list
    return


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
                    #term = Term(token, docID, index)

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
        # ritorno il dizionario con i termini ordinati in ordine alfabetico
        return idx



class IRsystem:
    def __init__(self, corpus, index):
        self._corpus = corpus
        self._index = index

    @classmethod
    def from_corpus(cls, corpus):
        index = InvertedIndex.from_corpus(corpus)
        #print(index)
        return cls(corpus, index)

    def answer_query(self, words, connections):
        '''da riabilitare'''
        #norm_words = map(normalize, words)

        norm_words = words
        postings = []

        for w in norm_words:
            try:
                res = find_prefix(self._index._root, w)
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
                    plist = reduce(lambda x, y: x.not_query(y), postings)
        return plist.get_from_corpus(self._corpus)
        #plist = reduce(lambda x, y: x.phrase(y), postings)
        return res.get_from_corpus(self._corpus)

    def answer_phrase_query(self, words):
        #da riabilitare
        #norm_words = map(normalize, words)
        norm_words = words
        postings = []
        for w in norm_words:
            try:
                res = find_prefix(self._index._root, w)
            except KeyError:
                #implementare il not found
                print("{} not found. Did you mean {}?")
                pass
            postings.append(res)
        #having all the postings now I have to check if they are near
        plist = reduce(lambda x, y: x.phrase(y), postings)
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

    #answer = ir.answer_phrase_query(terms)
    answer = ir.answer_query(terms,connections)
    print(len(answer))
    for doc in answer:
        print(doc)

#lettura file, creazione e salvataggio indice
def initialization():
    tic = time.perf_counter()
    corpus = read_data_descriptions()
    ir = IRsystem.from_corpus(corpus)
    with open("trie.pickle", "wb") as file_:
        pickle.dump(ir, file_, -1)
    toc = time.perf_counter()
    print(f"Index created in {toc - tic:0.4f} seconds")

def operate():
    print("Retrieving index...")
    ir = pickle.load(open("trie.pickle", "rb", -1))
    print("Index retrieved!")
    tic = time.perf_counter()
    query(ir, "cat or space")
    toc = time.perf_counter()
    print(f"Query performed in {toc - tic:0.4f} seconds")


if __name__ == "__main__":
    #initialization()
    operate()

