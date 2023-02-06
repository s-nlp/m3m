from transformers import Pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


class Seq2SeqPipeline(Pipeline):
    """Seq2SeqPipeline - HF Pipiline for generatng set of candidates for QA problem
    Working with ConditionalGeneration HF models
    """

    def _sanitize_parameters(self, **kwargs):
        forward_kwargs = {}
        if "num_beams" in kwargs:
            forward_kwargs["num_beams"] = kwargs.get("num_beams", 200)
        if "num_return_sequences" in kwargs:
            forward_kwargs["num_return_sequences"] = kwargs.get(
                "num_return_sequences", 200
            )
        if "num_beam_groups" in kwargs:
            forward_kwargs["num_beam_groups"] = kwargs.get("num_beam_groups", 20)
        if "diversity_penalty" in kwargs:
            forward_kwargs["diversity_penalty"] = kwargs.get("diversity_penalty", 0.1)
        return {}, forward_kwargs, {}

    def preprocess(self, input_):
        return self.tokenizer(
            input_,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )

    def _forward(
        self,
        input_tensors,
        num_beams=200,
        num_return_sequences=200,
        num_beam_groups=20,
        diversity_penalty=0.1,
    ):
        outputs = self.model.generate(
            input_tensors["input_ids"].to(self.device),
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
        )
        return outputs

    def postprocess(self, model_outputs):
        candidates = self.tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return list(dict.fromkeys(candidates))


def build_seq2seq_pipeline(model_path: str, device="cpu") -> Seq2SeqPipeline:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipeline = Seq2SeqPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return pipeline
