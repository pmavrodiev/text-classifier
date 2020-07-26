 curl localhost:5001/predict --data-binary @predict_paylaod.json -H "Content-Type: text/plain"
 
 
 curl localhost:5001/predict --data-binary @predict_paylaod.json.gz -H "Content-Type: application/gzip"
 
 curl localhost:5001/predict --data-binary @predict_paylaod.json.gz -H "Content-Type: application/gzip" -H "Accept-Encoding: gzip" > output.json.gz