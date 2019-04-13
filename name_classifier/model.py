import torch
import numpy as np
from allennlp.models import Model
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.predictors import Predictor
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.vocabulary import Vocabulary
from typing import Iterator, List, Dict

@Model.register('homework4-model')
class RNNClassifier(Model):
    def __init__(self, 
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
                  
        self.encoder = encoder    
        self.h2o = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))        
        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.NLLLoss()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        
    def forward(self,
                name: Dict[str, torch.Tensor],
                label: torch.Tensor = None ) -> Dict[str, torch.Tensor]:
            
        mask = get_text_field_mask(name)
        input_tensor = name['tokens']
        one_hot = torch.zeros(input_tensor.size()[0], input_tensor.size()[1], self.encoder.get_input_dim())
        for i,index in enumerate(input_tensor):
            for j, letter in enumerate(index):
                one_hot[i][j][letter] = 1
        encoder_out = self.encoder(one_hot, mask)

        predicted = self.h2o(encoder_out)
        
        output = {"log_soft": self.softmax(predicted)}
        if label is not None:
            self.accuracy(predicted, label)
            output["loss"] =  self.loss(output["log_soft"], label)

       
        return output
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}