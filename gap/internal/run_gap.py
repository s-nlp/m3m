import os
import torch

from transformers import BartTokenizer

from internal.modeling_gap import GAPBartForConditionalGeneration as GAP
from internal.modeling_gap_type import GAPBartForConditionalGeneration as GAP_Type

from internal.data_relations_as_nodes import GAPDataloader, EventDataset, WebNLGDataset
from internal.data_relations_as_nodes import get_t_emb_dim


class GAPEvaluator:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        # Initialize tokenizer
        self.tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)

        # Inference on the test set
        checkpoint = args.output_dir
        if args.type_encoding:
            t_emb_dim = get_t_emb_dim(args)
            self.model = GAP_Type.from_pretrained(checkpoint, t_emb_dim=t_emb_dim)
        else:
            self.model = GAP.from_pretrained(checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))
        self.model.eval()

    def run(self, input_graph):
        dev_dataset = WebNLGDataset(self.logger, self.args, input_graph, self.tokenizer, "val")
        dev_dataloader = GAPDataloader(self.args, dev_dataset, "dev")
        predictions = self.inference(
            self.model,
            dev_dataloader,
            self.tokenizer,
            self.args,
            self.logger
        )

        return predictions[0]

    def inference(self, model, dev_dataloader, tokenizer, args, logger, save_predictions=False):
        predictions = []
        for i, batch in enumerate(dev_dataloader):
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]
            outputs = model.generate(input_ids=batch[0],
                                     attention_mask=batch[1],
                                     input_node_ids=batch[4],
                                     node_length=batch[5],
                                     adj_matrix=batch[6],
                                     num_beams=args.num_beams,
                                     length_penalty=args.length_penalty,
                                     max_length=args.max_output_length,
                                     early_stopping=True,)
            # Convert ids to tokens
            for input_, output in zip(batch[0], outputs):
                pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=args.clean_up_spaces)
                predictions.append(pred.strip())

        # Save the generated results
        if save_predictions:
            save_path = os.path.join(args.output_dir, "{}predictions.txt".format(args.prefix))
            with open(save_path, "w") as f:
                for pred in predictions:
                    f.write(pred + '\n')
            logger.info("Saved prediction in {}".format(save_path))

        if args.dataset == "eventNarrative":
            data_ref = [[data_ele[1].strip()] for data_ele in dev_dataloader.dataset.data]
        else:
            data_ref = [data_ele['text'] for data_ele in dev_dataloader.dataset.data]
        assert len(predictions) == len(data_ref)
        return predictions
