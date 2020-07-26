 # Summary
 
 
 # Training
 
 ## Google Collab
 
 # Test trained models
 
 ## Google Collab
 
 ## Locally via docker-compose
 
 To test already trained models you can deploy a simple RESTful Server locally using Docker.
 
 ### Build 
 
 Requirements:
     
    docker 19.03.8
    docker-compose 1.25.5    
 
 Note, these are the versions I used, it is likely that slightly older ones will work too. So if you have Docker and
 docker-compose installed, simply give it a try. 
 
 To run the API server execute the following:
 
    docker-compose -f docker/build/docker-compose.yml up
 
 
This will start a Docker container configured to download an already trained model from my Google Drive.
This configuration is specified in the `MODEL_TO_LOAD` environmental variable in `docker-compose.yml`
 
The format of the variable `MODEL_TO_LOAD` is a simple json schema:

    "[<model-name:str>, <google-drive-id:str>]"
 
where `model-name` is the name of the trained model artifact and `google-drive-id` is the id of the 
model object in my google drive. Currently the following trained models have been created:

| model-name                                   | google-drve-id                    | description                                                                                                                                                                                                                                                                                                                                                                          |
|----------------------------------------------|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| distilbert-base-cased_n_epochs_4_mincount300 | 13q92ILaUlwNXSatVSZZ9MaH921A6snNS | 6-layer, 768-hidden, 12-heads, 65M parameters  This is the Distilled Bert model, which is  a compressed Bert model, trained via  knowledge distillation.   The model has been fine-tuned on the given dataset for 4 epochs. Training labels that occur fewer than 300 times have been pruned.  Source: https://github.com/huggingface/transformers/tree/master/examples/distillation |
| distilbert-base-cased_n_epochs_2_mincount300 | 13jA73V9_BawKyWAGb_h-CRXPSMMbv7H6 | Same as above, only trained for 2 epochs.                                                                                                                                                                                                                                                                                                                                            |
| bert-english-multi-cased                     | 13aOeDt2CpPEwobLe98NFBz-hjpN9JNXX | Original Bert cased model (12-layer, 768-hidden, 12-heads, 110M parameters)  fine-tuned for the given dataset                                                                                                                                                                                                                                                                        |


The default model configured in `docker-compose.yml` is `distilbert-base-cased_n_epochs_4_mincount300`. 
Edit the file if you want to test the other models, as well.

Once downloaded the model will be stored in the container path `/app/trained_models`, which is mapped to 
`docker/models/trained_models` on the host.

 ### Run 
 
Once the container has successfully started and the chosen model has been downloaded you can invoke two REST endpoints: 
`/predict` and `/predict_raw`. 

- **/predict**

    To test the supplied predict_payload.json you can use `curl` to send the json file as payload to the endpoint:

      curl localhost:5001/predict --data-binary @predict_paylaod.json -H "Content-Type: text/plain"

    If the test data is too large, you can optionally gzip the payload file:
 
      curl localhost:5001/predict --data-binary @predict_paylaod.json.gz -H "Content-Type: application/gzip"
    
    And to retrieve a gzipped response (make sure to redirect the output to a file, so that you don't mess up your terminal):
 
      curl localhost:5001/predict --data-binary @predict_paylaod.json.gz -H "Content-Type: application/gzip" -H "Accept-Encoding: gzip" > output.json.gz
    
    By default `/predict` returns the 4 most confident labels for a given document.
    You can supply an optional argument to change that, e.g. `/predict/2` will return the top 2 most confident labels.
    
- **/predict_raw**

    To facilitate easier testing of the trained models, you invoke this endpoint and supply a raw texts as payload.
    Simply pass a file, where each line corresponds to the raw text of one document.

      curl localhost:5001/predict_raw --data-binary @predict_paylaod.txt -H "Content-Type: text/plain"

    If the test data is too large, you can optionally gzip the payload file:
 
      curl localhost:5001/predict_raw --data-binary @predict_paylaod.txt.gz -H "Content-Type: application/gzip"
    
    And to retrieve a gzipped response (make sure to redirect the output to a file, so that you don't mess up your terminal):
 
      curl localhost:5001/predict_raw --data-binary @predict_paylaod.txt.gz -H "Content-Type: application/gzip" -H "Accept-Encoding: gzip" > output.json.gz
    
    By default `/predict_raw` returns the 4 most confident labels for a given document.
    You can supply an optional argument to change that, e.g. `/predict_raw/2` will return the top 2 most confident labels.