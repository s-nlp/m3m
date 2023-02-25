# A System for Answering Simple Questions in Multiple Languages

This repository contains source code of the Web demo for answering one-hop questions over knowledge graph. 

Link to the online demo: [https://kgqa-nlp-zh.skoltech.ru](https://kgqa-nlp-zh.skoltech.ru)

<img width="700" alt="paris" src="https://user-images.githubusercontent.com/1456830/221340570-bdb95079-d68c-44ba-ad0a-e6613a812963.png">

Question answering in natural language is a key method for fulfilling information needs of users and for learning. In this work, we focus on the most popular type of questions, namely simple questions, such as "What is capital of France?". They mention an entity, e.g. "France" that is one hop away from the answer entity in terms of the underlying knowledge graph (KG), i.e. "Paris". We present a multilingual Knowledge Graph Question Answering (KGQA) a novel method and a system that ranks answer candidates according to the proximity of question's text and graph embeddings. We conducted extensive experiments with several English and multilingual datasets and two KGs -- Freebase and Wikidata. We demonstrate that the proposed method compares favorably across different KG embeddings and languages compared to strong baseline systems including complex rule-based pipelines, search-based solutions, and seq2seq based QA models. We make the code and trained models of our solution publicly available to contribute further developments in the field of multilingual KGQA and foster applications of QA technology in multiple supported languages. 

## How to add additional pipeline?

* You can put all requred congifs to pipeline to app/config.py
* You can put all required scripts to app/kgqa 
* Create file with pipeline name in app/pipelines/pipeline_name.py and put fastAPI router for pipeline here. All pipelines route methods must reture `PipelineResponce` object. You can add your own responce class to app/models, but it should be inhered from `PipelineResponce`.
* Add router to app/main.py: `app.include_router(pipeline_name.router)` (Now your pipeline must be availabe at backend level. Check it on /docs)
* Add pipeline_name to app/templates/search_bar.html in pipeline selector
