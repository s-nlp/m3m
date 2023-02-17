from functools import lru_cache
from nltk.stem.porter import PorterStemmer

from .ner import NerToSentenceInsertion


class EntitiesSelection:
    def __init__(self, ner_model: NerToSentenceInsertion):
        self.stemmer = PorterStemmer()
        self.ner_model = ner_model

    def entities_selection(self, entities_list, mgenre_predicted_entities_list):
        final_preds = []

        for pred_text in mgenre_predicted_entities_list:
            labels = []
            try:
                _label, lang = pred_text.split(" >> ")
                if lang == "en":
                    labels.append(_label)
            except Exception as e:
                raise e

            if len(labels) > 0:
                for label in labels:
                    label = label.lower()
                    if self._check_label_fn(label, entities_list):
                        final_preds.append(label)

        return final_preds

    def __call__(self, entities_list, mgenre_predicted_entities_list):
        return self.entities_selection(entities_list, mgenre_predicted_entities_list)

    @lru_cache(maxsize=8192)
    def _label_format_fn(self, label):
        return " ".join(
            [self.stemmer.stem(str(token)) for token in self.ner_model(label)]
        )

    def _check_label_fn(self, label, entities_list):
        label = self._label_format_fn(label)
        for entity in entities_list:
            entity = self._label_format_fn(entity)
            if label == entity:
                return True
        return False
