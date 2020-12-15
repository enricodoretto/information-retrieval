import _pickle as pickle
from functools import reduce
import csv
import nltk
import time
from term_operations import Term
from term_operations import DataDescription
from term_operations import process
from term_operations import process_query
from trie import Trie
import sys
import gc


sys.setrecursionlimit(10000)  # necessary to save a large file with pickle
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

    # building the index starting from the corpus
    @classmethod
    def from_corpus(cls, corpus):
        trie = Trie()
        number_of_docs = 0

        for docID, document in enumerate(corpus):
            number_of_docs += 1
            tokens = process(document.description)
            for index, token in enumerate(tokens):
                # create the term at which each rotation of the word will point
                term = Term(token, docID, index)
                token = token + "$"
                # rotate the word for the wildcard
                for i in token:
                    trie.insert(token, term)
                    token = token[1:] + token[0]

            if docID % 1000 == 0 and docID != 0:
                print(str(docID), end='...')
                # enable this to limit the dimension of the index for testing
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
        query_words = query
        i = 0
        while i < len(query_words):

            # if the word is not a connection retrieve the posting list
            if query_words[i] != "and" and query_words[i] != "or" and query_words[i] != "not":

                # the idea is to replace the word in the query with its corresponding posting list
                query_words[i] = query_words[i] + '$'

                # check if it is a wildcard
                counter = query_words[i].count('#')
                if not counter == 0:
                    if counter == 1:
                        # if it's a simple wildcard rotate until # is at the end
                        while not query_words[i].endswith("#"):
                            query_words[i] = query_words[i][1:] + query_words[i][0]
                        query_words[i] = query_words[i][:-1]

                        singlewcard = self._index._trie.get_wildcard(query_words[i])
                        if not singlewcard:
                            return
                        # the posting list of the wildcard will be the union of all the posting lists corresponding to that wildcard
                        res_single_wildcardcard = reduce(lambda x, y: x.union(y), singlewcard)
                        query_words[i] = res_single_wildcardcard

                    elif counter > 1:
                        # multiple wildcard
                        # keep only the last part of the wildcard and perform the query on that
                        cards = query_words[i].rsplit('#', 1)
                        key1 = '#' + cards[1]
                        while not key1.endswith("#"):
                            key1 = key1[1:] + key1[0]
                        key = key1[:-1]

                        # generate the regex to check if a term corrisponding to the simple wcard corresponds also to the multiple wildcard
                        key1 = key1[:-1]
                        key1 = key1[:-1]
                        key2 = cards[0].replace("#", ".+")
                        pattern = key1 + "\$" + key2 + ".+"
                        multiplewcards = self._index._trie.get_multiple_wildcard(key, pattern)
                        if not multiplewcards:
                            return
                        res_multiple_wildcard = reduce(lambda x, y: x.union(y), multiplewcards)
                        query_words[i] = res_multiple_wildcard

                else:
                    # simple word not wildcard
                    res = self._index._trie.search(query_words[i])
                    query_words[i] = res
                    if not res:
                        return
                i += 1

            # it the word is a connection skip to the next one
            else:
                i += 1

        j = 0
        # query_words now will be like: [Posting List, connection, Posting List, Posting List, connection, Posting List]
        # the idea now is to resolve an operation replacing the first item of the operation with the result and delete
        # the other items: if we have [Posting List, connection, Posting List] the result will be [Posting List], where
        # the resulting Posting List is the result of the operation.
        # We do this until the length of query_words is != 1 or an operation results null

        # at first we answer phrase queries
        while j < len(query_words)-1:
            if not isinstance(query_words[j], str) and not isinstance(query_words[j+1], str):
                query_words[j] = query_words[j].phrase(query_words[j+1])
                if not query_words[j]:
                    return
                del query_words[j+1]
            else: j += 1

        j = 0
        # now we can resolve AND, OR, NOT, AND NOT queries
        while len(query_words) != 1:
            if not isinstance(query_words[j], str) and query_words[j+1] == "and" and not isinstance(query_words[j+2], str):
                query_words[j] = query_words[j].intersection(query_words[j+2])
                if not query_words[j]:
                    return
                del query_words[j + 1]
                del query_words[j + 1]

            elif not isinstance(query_words[j], str) and query_words[j+1] == "or" and not isinstance(query_words[j+2], str):
                query_words[j] = query_words[j].union(query_words[j+2])
                if not query_words[j]:
                    return
                del query_words[j + 1]
                del query_words[j + 1]

            elif not isinstance(query_words[j], str) and query_words[j+1] == "and" and query_words[j+2] == "not" and not isinstance(query_words[j+3], str):
                query_words[j] = query_words[j].not_query(query_words[j+3])
                if not query_words[j]:
                    return
                del query_words[j + 1]
                del query_words[j + 1]
                del query_words[j + 1]

            elif query_words[j] == "not" and not isinstance(query_words[j+1], str):
                query_words[j] = query_words[j+1].not_query_all_docs(self._index._number_of_docs)
                if not query_words[j]:
                    return
                del query_words[j + 1]

        # get the title of the movies in the resulting posting list
        return query_words[0].get_from_corpus(self._corpus)


def query(ir, text):
    terms_query = process_query(text)
    answer = ir.answer_query_full(terms_query)

    if not answer:
        print("No matching found")
    else:
        for doc in answer:
            print(doc)
        print(len(answer))


def initialization():
    # disable the garbage collector for performance improvement
    gc.disable()

    tic = time.perf_counter()
    corpus = read_data_descriptions()
    ir = IRsystem.from_corpus(corpus)
    tac = time.perf_counter()
    print(f"Index builded in {tac - tic:0.4f} seconds")
    file = open("data/trie_index.pickle", "wb")
    pickle.dump(ir, file, protocol=-1)
    file.close()
    toc = time.perf_counter()
    print(f"Index saved in {toc - tac:0.4f} seconds")

    gc.enable()


def operate():
    ir = IRsystem
    while 1:
        op = input("Press 'i' to retrieve the index, 'q' to perform the query, any other key to quit: ")
        if op == "i":

            gc.disable()
            print("Retrieving index...")
            tic = time.perf_counter()
            f = open('data/trie_index.pickle', 'rb')

            ir = pickle.load(f)
            f.close()
            tac = time.perf_counter()
            print(f"Index retrieved in {tac - tic:0.4f} seconds")
            gc.enable()

        elif op == "q":
            que = input("Write the query: ")
            tac = time.perf_counter()
            query(ir, que)
            toc = time.perf_counter()
            print(f"Query performed in {toc - tac:0.4f} seconds")

        else:
            quit()


if __name__ == "__main__":
    op = input("Press 'i' to create a new index, 'q' to perform a query, any other key to quit: ")
    if op == "i":
        initialization()
    elif op == "q":
        operate()
    else:
        quit()
