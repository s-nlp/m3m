# QA Prod Demo

## How to add additional pipeline?

* You can put all requred congifs to pipeline to app/config.py
* You can put all required scripts to app/kgqa 
* Create file with pipeline name in app/pipelines/pipeline_name.py and put fastAPI router for pipeline here. All pipelines route methods must reture `PipelineResponce` object. You can add your own responce class to app/models, but it should be inhered from `PipelineResponce`.
* Add router to app/main.py: `app.include_router(pipeline_name.router)` (Now your pipeline must be availabe at backend level. Check it on /docs)
* Add pipeline_name to app/templates/search_bar.html in pipeline selector