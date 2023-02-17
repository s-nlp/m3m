"""Module for NER insertion to the sentence"""

import spacy


class NerToSentenceInsertion:
    """Module for adding START and END tokens"""

    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.model = spacy.load(model_path)

    def entity_labeling(self, test_question, return_entities_list=False):
        """First lettters capitalization and START/END tokens for entities insertion"""

        # ner part
        nlp = self.model
        doc = nlp(test_question)
        entities_list = [ent.text for ent in doc.ents]
        num_entities = len(entities_list)
        if num_entities > 0:
            ner_question = test_question
            for entity in entities_list:
                entity_index_in_string = ner_question.find(entity)
                ner_question = (
                    ner_question[:entity_index_in_string]
                    + " [START] "
                    + ner_question[
                        entity_index_in_string : entity_index_in_string + len(entity)
                    ]
                    + " [END] "
                    + ner_question[entity_index_in_string + len(entity) :]
                ).replace("  ", " ")
        else:
            ner_question = "[START] " + str(test_question) + " [END]"

        # LargeCase part
        sent_split = []
        for elem in ner_question.split(" "):
            if elem != "":
                sent_split.append(elem[0].upper() + elem[1:])
        ner_largecase_question = " ".join(sent_split)

        if return_entities_list:
            return ner_largecase_question, entities_list
        return ner_largecase_question

    def __call__(self, test_question, return_entities_list=False):
        return self.entity_labeling(test_question, return_entities_list)
