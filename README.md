# A System for Answering Simple Questions in Multiple Languages

This repository contains source code of the Web demo for answering one-hop questions over knowledge graph. 

Link to the online demo: [https://kgqa-nlp-zh.skoltech.ru](https://kgqa-nlp-zh.skoltech.ru)

<img width="700" alt="paris" src="https://user-images.githubusercontent.com/1456830/221340570-bdb95079-d68c-44ba-ad0a-e6613a812963.png">

## How to add additional pipeline?

* You can put all requred congifs to pipeline to app/config.py
* You can put all required scripts to app/kgqa 
* Create file with pipeline name in app/pipelines/pipeline_name.py and put fastAPI router for pipeline here. All pipelines route methods must reture `PipelineResponce` object. You can add your own responce class to app/models, but it should be inhered from `PipelineResponce`.
* Add router to app/main.py: `app.include_router(pipeline_name.router)` (Now your pipeline must be availabe at backend level. Check it on /docs)
* Add pipeline_name to app/templates/search_bar.html in pipeline selector
