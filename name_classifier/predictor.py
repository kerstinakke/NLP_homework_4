import numpy as np
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

#@Predictor.register('homework4-predictor')
class PaperClassifierPredictor(Predictor):
    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        name = json_dict['name']
        instance = self._dataset_reader.name_to_instance(name)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        prediction =  self.predict_instance(instance)
        print(prediction)
        prediction['all_labels'] = all_labels
        prediction['pred_label'] = all_labels[np.argmax(prediction['log_soft'])]

        return prediction