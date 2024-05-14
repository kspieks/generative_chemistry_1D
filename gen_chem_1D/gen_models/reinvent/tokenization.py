import re

import numpy as np

SMI_REGEX_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"

class RegexTokenizer:
    """
    Class for tokenizing SMILES strings using a regular expression.
    Adapted from https://github.com/rxn4chemistry/rxnfp.

    Args:
        regex_pattern: regex pattern used for tokenization.
    """

    def __init__(self, regex_pattern=SMI_REGEX_PATTERN):
        self.regex_pattern = regex_pattern
        self.regex = re.compile(self.regex_pattern)

    def tokenize(self, smiles):
        """
        Performs the regex tokenization.
        
        Args:
            smiles: smiles to tokenize.
        
        Returns:
            List of extracted tokens.
        """
        tokens = [token for token in self.regex.findall(smiles)]
        return tokens


class Vocabulary(object):
    """Class to encode/decode from SMILES to an array of indices"""
    def __init__(self,
                 init_from_file=None,
                 max_length=140,
                 ):
        self.special_tokens = ['EOS', 'GO']
        self.additional_chars = list()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[NH]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['EOS']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES string and return a list of tokens"""
        tokenizer = RegexTokenizer()
        tokenized = tokenizer.tokenize(smiles)
        tokenized.append('EOS')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.append(char)
        char_list = list(self.additional_chars)
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return f"Vocabulary containing {len(self)} tokens: {self.chars}"


def create_vocabulary(smiles_list):
    """Returns all tokens present in a list of SMILES"""
    tokenizer = RegexTokenizer()
    vocab_tokens = set()
    for smi in smiles_list:
        token_list = tokenizer.tokenize(smi)
        vocab_tokens.update(token_list)

    # ensure that all numbers are in vocabulary
    numbers = [str(i) for i in range(0, 10)]
    vocab_tokens.update(numbers)
    
    return vocab_tokens
