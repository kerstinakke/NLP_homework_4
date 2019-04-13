import glob
import os
import unicodedata
import string

from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

@DatasetReader.register('homework4-reader')
class NameDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.all_letters = string.ascii_letters + " .,;'"

    
    def unicodeToAscii(self,s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )
    
    # Read a file and split into lines
    def readLines(self,filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [self.unicodeToAscii(line) for line in lines]
    
    def name_to_instance(self, name:str, category: str = None) -> Instance:
        name_field = TextField([Token(char) for char in name], self.token_indexers)
        fields = {"name": name_field}

        if category:
            label_field = LabelField(label=category)
            fields["label"] = label_field

        return Instance(fields)
    
    def _read(self, dir_path: str) -> Iterator[Instance]:
        for filename in glob.glob((os.path.join(dir_path,'*.txt'))):
            category = os.path.splitext(os.path.basename(filename))[0]
            lines = self.readLines(filename)
            for name in lines:
                yield self.name_to_instance(name, category)
        