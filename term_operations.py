import re
from functools import total_ordering, reduce
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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


def normalize(text):
    no_punctuation = re.sub(r'[^\w^\s^#-]', '', text)
    #no_punctuation = re.sub(r'[^\P{P}-#]+', '', text)
    downcase = no_punctuation.lower()
    return downcase


def process(desc):
    stop_words = set(stopwords.words('english'))

    normalized = normalize(desc)
    tokenized = normalized.split()
    no_stop_words = [w for w in tokenized if not w in stop_words]
    processed = stem(no_stop_words)
    return processed

def stem(text):
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in text]
    return stemmed


def tokenize_query(query):


    normalized = normalize(query)
    tokenized = normalized.split()
    stemmed = [ps.stem(word) for word in tokenized]
    return stemmed

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
                        break
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
