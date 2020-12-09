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

    # scorro l'albero fino a trovare la wildcard
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

        self.scan_wildcard_child(node, temp_word)
        return self.word_list

    # una volta trovata la wildcard scorro i figli ricorsivamente e ritorno le posting list
    def scan_wildcard_child(self, node, word):
        if node.last:
            self.word_list.append(node.term.posting_list)

        for a, n in node.children.items():
            self.scan_wildcard_child(n, word + a)

    # unica differenza dalla wildcard normale è che passo direttamente la regex della multiple
    # e verifico direttamente se è il termine che soddisfa la wildcard semplice è da ritornare o no
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

        self.scan_multiple_wildcard_child(node, temp_word, init)
        return self.word_list

    def scan_multiple_wildcard_child(self, node, word, init):
        reg = re.compile(init)

        if node.last and re.match(reg, word):
            self.word_list.append(node.term.posting_list)

        for a, n in node.children.items():
            self.scan_multiple_wildcard_child(n, word + a, init)