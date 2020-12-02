import re
from term_operations import Term
from term_operations import PostingList




class TrieNode():
    def __init__(self):
        # Initialising one node for trie
        self.children = {}
        self.last = False
        self.posting_list = PostingList
        self.term = Term


class Trie():
    def __init__(self):

        # Initialising the trie structure.
        self.root = TrieNode()
        self.word_list = []


    def insert(self, key, term):

        # Inserts a key into trie if it does not exist already.
        # And if the key is a prefix of the trie node, just
        # marks it as leaf node.
        node = self.root

        for a in list(key):
            if not node.children.get(a):
                node.children[a] = TrieNode()

            node = node.children[a]

        node.last = True
        try:
            node.term.merge(term)
            #node.posting_list.merge(PostingList.from_docID(docID, position))
        except TypeError:
            node.term = term
            #node.posting_list = PostingList.from_docID(docID, position)


    def search(self, key):

        # Searches the given key in trie for a full match
        # and returns True on success else returns False.
        node = self.root
        found = True

        for a in list(key):
            if not node.children.get(a):
                found = False
                break

            node = node.children[a]

        if node.last:
            return node.term.posting_list


    def suggestionsRec(self, node, word):

        # Method to recursively traverse the trie
        # and return a whole word.
        if node.last:
            self.word_list.append(node.term.posting_list)

        for a, n in node.children.items():
            self.suggestionsRec(n, word + a)

    def getWildcard(self, key):
        self.word_list = []
        # Returns all the words in the trie whose common
        # prefix is the given key thus listing out all
        # the suggestions for autocomplete.
        node = self.root
        not_found = False
        temp_word = ''

        for a in list(key):
            if not node.children.get(a):
                not_found = True
                break

            temp_word += a
            node = node.children[a]

        if not_found:
            return 0
        elif node.last and not node.children:
            return -1

        self.suggestionsRec(node, temp_word)

        return self.word_list


    def suggestionsRecMW(self, node, word, init):

        # Method to recursively traverse the trie
        # and return a whole word.
        reg = re.compile(init)

        if node.last and re.match(reg, word):
            self.word_list.append(node.term.posting_list)

        for a, n in node.children.items():
            self.suggestionsRecMW(n, word + a, init)

    def getWildcardMW(self, key, init):
        self.word_list = []

        # Returns all the words in the trie whose common
        # prefix is the given key thus listing out all
        # the suggestions for autocomplete.
        node = self.root
        not_found = False
        temp_word = ''

        for a in list(key):
            if not node.children.get(a):
                not_found = True
                break

            temp_word += a
            node = node.children[a]

        if not_found:
            return 0
        elif node.last and not node.children:
            return -1

        self.suggestionsRecMW(node, temp_word, init)

        return self.word_list


