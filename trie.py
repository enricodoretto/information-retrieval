import re
from term_operations import Term
from term_operations import PostingList


class TrieNode:
    def __init__(self):
        # Initialising one node for trie
        self.children = {}
        self.last = False
        self.posting_list = PostingList
        self.term = Term


class Trie:
    def __init__(self):
        self.root = TrieNode()
        # tenere traccia della parola corrente durante la ricerca
        self.word_list = []

    def insert(self, key, term):
        node = self.root
        for a in list(key):
            if not node.children.get(a):
                node.children[a] = TrieNode()

            node = node.children[a]

        node.last = True
        try:
            # se il nodo è già presente faccio il merge delle rispettive posting list
            node.term.merge(term)
        except TypeError:
            # se il nodo non esiste già
            node.term = term

    def search(self, key):
        # ricerca della parola non wildcard
        node = self.root
        found = True
        for a in list(key):
            if not node.children.get(a):
                found = False
                break

            node = node.children[a]

        if node.last:
            return node.term.posting_list

    def suggestions(self, node, word):
        if node.last:
            self.word_list.append(node.term.posting_list)

        for a, n in node.children.items():
            self.suggestions(n, word + a)

    def get_wildcard(self, key):
        self.word_list = []

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

        self.suggestions(node, temp_word)

        return self.word_list

    def suggestions_multiple_wildcard(self, node, word, init):
        reg = re.compile(init)

        if node.last and re.match(reg, word):
            self.word_list.append(node.term.posting_list)

        for a, n in node.children.items():
            self.suggestions_multiple_wildcard(n, word + a, init)

    def get_multiple_wildcard(self, key, init):
        self.word_list = []

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

        self.suggestions_multiple_wildcard(node, temp_word, init)

        return self.word_list
