import re
from term_operations import Term


class TrieNode:
    def __init__(self):
        self.children = {}
        self.last = False
        self.term = Term


class Trie:
    def __init__(self):
        self.root = TrieNode()
        # keep track of the current word during the research
        self.word_list = []

    def insert(self, token, term):
        node = self.root
        for a in list(token):
            if not node.children.get(a):
                node.children[a] = TrieNode()
            node = node.children[a]
        node.last = True
        try:
            # if the node is already present merge the posting lists
            node.term.merge(term)
        except TypeError:
            # if the node does not exist
            node.term = term

    def search(self, term):
        # search of not wildcard word
        node = self.root
        found = True
        for a in list(term):
            if not node.children.get(a):
                found = False
                break
            node = node.children[a]
        if node.last:
            return node.term.posting_list

    # scan the tree until the wildcard is found, then need to return all the children
    def get_wildcard(self, wcard):
        self.word_list = []
        node = self.root
        not_found = False
        temp_word = ''
        for a in list(wcard):
            if not node.children.get(a):
                not_found = True
                break
            temp_word += a
            node = node.children[a]
        if not_found:
            return
        elif node.last and not node.children:
            return
        self.scan_wildcard_child(node, temp_word)
        return self.word_list

    # once the wildcard is found scan recursively its children and return their posting lists
    def scan_wildcard_child(self, node, word):
        if node.last:
            self.word_list.append(node.term.posting_list)
        for a, n in node.children.items():
            self.scan_wildcard_child(n, word + a)

    # the only difference with the single wildcard is that with the multiple wildcard we need to pass the regex
    # to check if the child of the simple wildcard matches the original query
    def get_multiple_wildcard(self, wcard, init):
        self.word_list = []
        node = self.root
        not_found = False
        temp_word = ''
        for a in list(wcard):
            if not node.children.get(a):
                not_found = True
                break

            temp_word += a
            node = node.children[a]
        if not_found:
            return
        elif node.last and not node.children:
            return

        self.scan_multiple_wildcard_child(node, temp_word, init)
        return self.word_list

    def scan_multiple_wildcard_child(self, node, word, init):
        reg = re.compile(init)
        # the posting list of a child of the wildcard is returned only if the term matches the original query
        if node.last and re.match(reg, word):
            self.word_list.append(node.term.posting_list)

        for a, n in node.children.items():
            self.scan_multiple_wildcard_child(n, word + a, init)