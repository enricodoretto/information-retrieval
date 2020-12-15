import re
from functools import total_ordering
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# normalization is done removing punctuation and setting to lower case every word
def normalize(text):
    no_punctuation = re.sub(r'[^\w^\s^#-]', '', text)
    downcase = no_punctuation.lower()
    return downcase


# stemming is done using the PorterStemmer
# don't want to stem "and, or, not" because are used for the query
def stem(text):
    ps = PorterStemmer()
    stemmed = [ps.stem(word) for word in text if word != "and" or word != "or" or word != "not"]
    return stemmed

# process is the main method for processing a movie description
# first normalization is done, than tokenize by splitting by space, simple but effective
# remove the stop words and stem at the end
def process(desc):
    stop_words = set(stopwords.words('english'))

    normalized = normalize(desc)
    tokenized = normalized.split()
    no_stop_words = [w for w in tokenized if not w in stop_words]
    processed = stem(no_stop_words)
    return processed


# the process of a query is the same as the description, the only difference is that in removing stop words
# "and, or, not" are keeped
def process_query(query):
    stop_words = set(stopwords.words('english'))

    normalized = normalize(query)
    tokenized = normalized.split()
    no_stop_words = []
    for w in tokenized:
        if w == "and" or w == "or" or w == "not" or w not in stop_words:
            no_stop_words.append(w)

    stemmed = stem(no_stop_words)
    return stemmed


class ImpossibleMergeError(Exception):
    pass


@total_ordering
class Term:

    # with a new term the posting list contains only the docID of the first movie
    def __init__(self, term, docID, position):
        self.term = term
        self.posting_list = PostingList.from_docID(docID, position)

    # merging the posting list of the terms if two terms are the same
    def merge(self, other):
        if self.term == other.term:
            self.posting_list.merge(other.posting_list)
        else:
            # a new term node in the trie needs to be created
            raise ImpossibleMergeError()

    def __eq__(self, other):
        return self.term == other.term

    def __gt__(self, other):
        return self.term > other.term

    def __repr__(self):
        return self.term + ": " + repr(self.posting_list)


@total_ordering
class Posting:

    # a posting is formed by a docID and a list of positions of the term in the document
    def __init__(self, docID, position):
        self._docID = docID
        self._poslist = []
        self._poslist.append(position)

    def get_from_corpus(self, corpus):
        return corpus[self._docID].title

    def __eq__(self, other):
        return self._docID == other._docID

    def __gt__(self, other):
        # to keep the order in the posting
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
        plist = cls()
        plist._postings = [Posting(docID, position)]
        return plist

    @classmethod
    def from_posting_list(cls, postingList):
        plist = cls()
        plist._postings = postingList
        return plist

    def merge(self, other):
        # if a term is present different times in a document add to the posting list the position of the term repeated.
        # the posting lists are supposed to be ordered increasingly
        i = 0
        last = self._postings[-1]
        while i < len(other._postings) and last == other._postings[i]:
            last._poslist.append(other._postings[i]._poslist[0])
            i += 1
        self._postings += other._postings[i:]

    def intersection(self, other):
        # intersect the posting list of two terms to answer the query AND
        intersection = []
        i = 0
        j = 0
        while i < len(self._postings) and j < len(other._postings):
            if self._postings[i] == other._postings[j]:
                # add the posting list to the result only if they are the same document
                intersection.append(self._postings[i])
                i += 1
                j += 1
            # else move forward the posting list that is currently lower
            elif self._postings[i] < other._postings[j]:
                i += 1
            else:
                j += 1
        # create a new Posting List to return as result
        return PostingList.from_posting_list(intersection)

    def union(self, other):
        # perform the union of two terms to answer the query OR
        union = []
        i = 0
        j = 0
        while i < len(self._postings) and j < len(other._postings):
            # add the posting list to the result in any case and move forward the posting list that is lower
            if self._postings[i] == other._postings[j]:
                union.append(self._postings[i])
                i += 1
                j += 1
            elif self._postings[i] < other._postings[j]:
                union.append(self._postings[i])
                i += 1
            else:
                union.append(other._postings[j])
                j += 1
        # if we arrived at the end of a posting list we need to add the remaining elements of the other one
        for k in range(i, len(self._postings)):
            union.append(self._postings[k])
        for k in range(j, len(other._postings)):
            union.append(other._postings[k])
        return PostingList.from_posting_list(union)

    def not_query(self, other):
        # add the posting list only if it's present in other but not in self
        not_term = []
        i = 0
        j = 0
        while i < len(self._postings) and j < len(other._postings):
            if self._postings[i] == other._postings[j]:
                # if the same not add
                i += 1
                j += 1
            elif self._postings[i] < other._postings[j]:
                not_term.append(self._postings[i])
                i += 1
            else:
                j += 1
        return PostingList.from_posting_list(not_term)

    def not_query_all_docs(self, number_of_docs):
        # to answer a not query on all docs the number of docs is needed
        # than we create a fake posting list adding every document, since we know the number of docs and the docIDs are
        # number from 1 to number_of_docs is easy, than we remove the posting if it's contained in the posting list of
        # the term in the not query
        alldocs = []
        docID = 1
        # generate the false posting list
        while docID < number_of_docs:
            alldocs.append(Posting(docID, 1))
            docID += 1
        i = 0
        # remove from the fake posting list the postings in the self posting list
        while i < len(self._postings):
            alldocs.remove(self._postings[i])
            i += 1

        return PostingList.from_posting_list(alldocs)

    # phrase query is similar to the intersection method, the difference is that if the two terms are in the same
    # document we scan the positions and return that posting only if the terms are near
    def phrase(self, other):
        phrase = []
        i = 0
        j = 0
        while i < len(self._postings) and j < len(other._postings):
            x = 0
            y = 0
            if self._postings[i] == other._postings[j]:
                # in the same document, scan the position list to check if they are near
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

    def get_from_corpus(self, corpus):
        return list(map(lambda x: x.get_from_corpus(corpus), self._postings))

    def __len__(self):
        return len(self._postings)

    def __iter__(self):
        return iter(self._postings)

    def __repr__(self):
        return ", ".join(map(str, self._postings))


class DataDescription:
    # couple title - description for the corpus
    def __init__(self, title, description):
        self.title = title
        self.description = description
