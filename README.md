# A System for Answering Simple Questions in Multiple Languages

This repository contains source code of the Web demo for answering one-hop questions over knowledge graph. 

- Link to the online demo: [https://kgqa-nlp-zh.skoltech.ru](https://kgqa-nlp-zh.skoltech.ru)

- Link to the video presentation of the demo: [https://www.youtube.com/watch?v=r5eM6kTYPcQ](https://www.youtube.com/watch?v=r5eM6kTYPcQ)

<img width="700" alt="paris" src="https://user-images.githubusercontent.com/1456830/221340570-bdb95079-d68c-44ba-ad0a-e6613a812963.png">

Question answering in natural language is a key method for fulfilling information needs of users and for learning. In this work, we focus on the most popular type of questions, namely simple questions, such as "What is capital of France?". They mention an entity, e.g. "France" that is one hop away from the answer entity in terms of the underlying knowledge graph (KG), i.e. "Paris". We present a multilingual Knowledge Graph Question Answering (KGQA) a novel method and a system that ranks answer candidates according to the proximity of question's text and graph embeddings. We conducted extensive experiments with several English and multilingual datasets and two KGs -- Freebase and Wikidata. We demonstrate that the proposed method compares favorably across different KG embeddings and languages compared to strong baseline systems including complex rule-based pipelines, search-based solutions, and seq2seq based QA models. We make the code and trained models of our solution publicly available to contribute further developments in the field of multilingual KGQA and foster applications of QA technology in multiple supported languages. 

## How to add additional pipeline?

* You can put all requred congifs to pipeline to `app/config.py`
* You can put all required scripts to `app/kgqa` 
* Create file with pipeline name in `app/pipelines/pipeline_name.py` and put fastAPI router for pipeline here. All pipelines route methods must reture `PipelineResponce` object. You can add your own responce class to `app/models`, but it should be inhered from `PipelineResponce`.
* Add router to `app/main.py`: `app.include_router(pipeline_name.router)` (Now your pipeline must be availabe at backend level. Check it on /docs)
* Add pipeline_name to app/templates/search_bar.html in pipeline selector


## Citation

```
@inproceedings{razzhigaev-etal-2023-system,
    title = "A System for Answering Simple Questions in Multiple Languages",
    author = "Razzhigaev, Anton  and
      Salnikov, Mikhail  and
      Malykh, Valentin  and
      Braslavski, Pavel  and
      Panchenko, Alexander",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-demo.51",
    pages = "524--537",
    abstract = "Our research focuses on the most prevalent type of queries{---} simple questions {---}exemplified by questions like {``}What is the capital of France?{''}. These questions reference an entity such as {``}France{''}, which is directly connected (one hop) to the answer entity {``}Paris{''} in the underlying knowledge graph (KG). We propose a multilingual Knowledge Graph Question Answering (KGQA) technique that orders potential responses based on the distance between the question{'}s text embeddings and the answer{'}s graph embeddings. A system incorporating this novel method is also described in our work.Through comprehensive experimentation using various English and multilingual datasets and two KGs {---} Freebase and Wikidata {---} we illustrate the comparative advantage of the proposed method across diverse KG embeddings and languages. This edge is apparent even against robust baseline systems, including seq2seq QA models, search-based solutions and intricate rule-based pipelines. Interestingly, our research underscores that even advanced AI systems like ChatGPT encounter difficulties when tasked with answering simple questions. This finding emphasizes the relevance and effectiveness of our approach, which consistently outperforms such systems. We are making the source code and trained models from our study publicly accessible to promote further advancements in multilingual KGQA.",
}
```

## Contacts

If you find some issue, do not hesitate to add it to [Github Issues](https://github.com/skoltech-nlp/m3m/issues).

For any questions and the **test part** of the data, please contact: Anton Razzhigaev (anton.razzhigaev@skol.tech)
