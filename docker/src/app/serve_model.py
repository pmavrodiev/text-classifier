import gzip
import pandas as pd
import gdown
import json
import logging
import os
import zipfile

from json import JSONDecodeError
from io import StringIO, BytesIO
from farm.infer import Inferencer
from flask import Flask
from flask import request
from flask import Response
from flask_script import Manager, Command, Option
from gunicorn.app.base import Application
from utils import Document
from simple_logging.custom_logging import setup_custom_logger
from optparse import OptionParser


# -------------------------------------
# Set up the app
# -------------------------------------
app = Flask(__name__)
# we keep the active model used by the /predict* endpoints in this variable
model_name = None

# -------------------------------------
# Set up logger
# -------------------------------------
LOGGING_LEVEL = logging.DEBUG
log_file = os.path.join('./log', 'web_server.log')

app.logger = setup_custom_logger('FLASK_WEB_SERVER', LOGGING_LEVEL, flog=log_file)
app.logger.newline()

app.logger.info("App is initializing")


def get_predictions(model_path, docs_to_predict, top_n=4):
    """
    This is a convenience function, which takes a path to a trained FARM model and a list of Documents and
    returns the predictions of the model for each Document.

    The model must exist locally at the specified path.
    :param model_path: A path to a locally stored FARM model
    :param docs_to_predict: A list of Documents to predict
    :param top_n: Return the top N predictions ranked according to confidence (default 4)
    :return: A list of lists of the format [ [doc_1], [doc_2], ..., [doc_N]],
             where [doc_X] = [ [<predicted_label_1>, <confidence>],..., [[<predicted_label_M>, <confidence>]] ]
             we return as many predicted labels as requested from top_n
    """
    nlp = Inferencer.load(model_path, task_type="text_classification")
    # get the labels used for training from the FARM Processor
    label_list = nlp.processor.tasks['text_classification']['label_list']

    # rename the labels to their original format, i.e replace "-" with ","
    label_list = [label.replace("-", ",") for label in label_list]

    to_predict = [{"text": doc.get_text()} for doc in docs_to_predict]
    result = nlp.inference_from_dicts(to_predict)

    # we now just have to loop through the 'result' structure and format the expected output accordingly
    output_list = []
    for inference_sample in result:
        predictions = inference_sample['predictions']
        for pred in predictions:
            predictions_df = pd.DataFrame({"labels": label_list,
                                           "predicted_proba": pred['probability']}).sort_values(by="predicted_proba",
                                                                                                ascending=False).head(
                top_n)
            sample_output = [[label, confidence] for label, confidence in zip(predictions_df["labels"],
                                                                              predictions_df['predicted_proba'])]
            output_list.append(sample_output)
    return output_list


@app.route('/healthz', methods=['GET'])
def health():
    """
    Just a simple /healthz API in case this ever gets into a K8s cluster
    :return:
    """
    app.logger.info("Got a GET for /healthz")
    return "I am healthy. Served by worker: " + str(os.getpid())


@app.route('/predict_raw', methods=['POST'])
@app.route('/predict_raw/<int:how_many>', methods=['POST'])
def parse_request_raw(how_many=4):
    """
    A convenience endpoint to predict on raw texts.

    This function takes as a payload string data (as supplied in a file by curl for example)
    where each line is the actual raw text of a Document.

    The payload data can optionally be gzipped, in which case the request must contain the header
    "Content-Type: application/gzip". Otherwise use "Content-Type: text/plain"

    The endpoint also optionally returns gzipped data. Specify "Accept-Encoding: gzip" to enable that.

    Note that no preprocessing will be done on the texts, except for new-line and tabs removal.
    The purpose of this function is to easily test the model on new texts, without the need to package them in the
    expected json schema.
    By default we return the top 4 most confident labels for each Document, unless the endpoint is invoked
    with the optional 'how_many' parameter

    :return: A list of lists of the format [ [doc_1], [doc_2], ..., [doc_N]],
             where [doc_X] = [ [<predicted_label_1>, <confidence>],..., [[<predicted_label_M>, <confidence>]] ]
             we return as many predicted labels as requested from top_n
    """

    app.logger.info("Got a POST for /predict_raw")

    global model_name

    if model_name is None:
        app.logger.error("No model configured yet")
        return Response("{'Messsage':'No model configured yet'}",
                        status=500, mimetype='text/plain')

    if len(request.data) == 0:
        return Response("{'Messsage':'No data in request'}",
                        status=400, mimetype='text/plain')

    if request.content_type == "text/plain":
        # we got a plain text
        uncompressed_data = request.data
    elif request.content_type == "application/gzip":
        # we got gzipped data
        compressed_data = BytesIO(request.data)
        uncompressed_data = gzip.GzipFile(fileobj=compressed_data, mode='r').read()
    else:
        return Response("{'Messsage':'Specify Content-Type in request header. "
                        "One of 'text/plain' or 'application/gzip'}",
                        status=400, mimetype='text/plain')

    # parse the payload
    try:
        f = StringIO(uncompressed_data.decode("utf-8"))
        predict_documents = []
        for doc in f:
            # skip empty lines
            if doc.strip() != "":
                d = Document(content=doc)
                predict_documents.append(d)
    except Exception as ex:
        app.logger.error(ex)

    model_path = os.path.join("/app/trained_models/", model_name,
                              "content/trained_models", model_name)
    output_list = get_predictions(model_path, predict_documents, top_n=how_many)

    # finally return
    # return type depends on what the client supports
    if 'gzip' in request.headers.get('Accept-Encoding', ''):
        # yes, we will return gzipped data
        gzip_buffer = BytesIO()
        gzip_file = gzip.GzipFile(mode='wb', fileobj=gzip_buffer)

        gzip_file.write(str(output_list).encode('utf-8'))
        gzip_file.close()

        app.logger.info("Accepts gzip, will return gzipped data")
        response = Response(gzip_buffer.getvalue(), status=200,
                            mimetype='application/gzip')
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Vary'] = 'Accept-Encoding'
        response.headers['Content-Length'] = len(response.data)
    else:
        # no, return plain text
        response = Response(str(output_list).encode('utf-8'), status=200,
                            mimetype='text/plain')
        response.headers['Content-Length'] = len(response.data)

    return response


@app.route('/predict', methods=['POST'])
@app.route('/predict/<int:how_many>', methods=['POST'])
def parse_request(how_many=4):
    """
    This is the main predict endpoints that can be used on json data formatted according to the schema in the challenge.

    It expects the json-formatted data in the payload of the request.
    The payload data can optionally be gzipped, in which case the request must contain the header
    "Content-Type: application/gzip".
    Otherwise use "Content-Type: text/plain"

    The endpoint also optionally returns gzipped data. Specify "Accept-Encoding: gzip" to enable that.

    By default we return the top 4 most confident labels for each Document, unless the endpoint is invoked
    with the optional 'how_many' parameter

    :return: A list of lists of the format [ [doc_1], [doc_2], ..., [doc_N]],
             where [doc_X] = [ [<predicted_label_1>, <confidence>],..., [[<predicted_label_M>, <confidence>]] ]
             we return as many predicted labels as requested from top_n
    """

    app.logger.info("Got a POST for /predict")

    global model_name

    if model_name is None:
        app.logger.error("No model configured yet")
        return Response("{'Messsage':'No model configured yet'}",
                        status=500, mimetype='text/plain')

    if len(request.data) == 0:
        return Response("{'Messsage':'No data in request'}",
                        status=400, mimetype='text/plain')

    if request.content_type == "text/plain":
        # we got a plain text
        uncompressed_data = request.data
    elif request.content_type == "application/gzip":
        # we got gzipped data
        compressed_data = BytesIO(request.data)
        uncompressed_data = gzip.GzipFile(fileobj=compressed_data, mode='r').read()
    else:
        return Response("{'Messsage':'Specify Content-Type in request header. "
                        "One of 'text/plain' or 'application/gzip'}",
                        status=400, mimetype='text/plain')

    # parse the payload
    try:
        json_corpus = json.loads(uncompressed_data)
    except JSONDecodeError as jde:
        app.logger.error(jde)
        return Response("{'Messsage':'Malformatted data'}",
                        status=400, mimetype='text/plain')

    predict_documents = []
    for doc in json_corpus:
        d = Document(doc['metadata'], doc['content'])
        predict_documents.append(d)

    model_path = os.path.join("/app/trained_models/", model_name,
                              "content/trained_models", model_name)
    output_list = get_predictions(model_path, predict_documents, top_n=how_many)

    # finally return
    # return type depends on what the client supports
    if 'gzip' in request.headers.get('Accept-Encoding', ''):
        # yes, we will return gzipped data
        gzip_buffer = BytesIO()
        gzip_file = gzip.GzipFile(mode='wb', fileobj=gzip_buffer)

        gzip_file.write(str(output_list).encode('utf-8'))
        gzip_file.close()

        app.logger.info("Accepts gzip, will return gzipped data")
        response = Response(gzip_buffer.getvalue(), status=200,
                            mimetype='application/gzip')
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Vary'] = 'Accept-Encoding'
        response.headers['Content-Length'] = len(response.data)
    else:
        # no, return plain text
        response = Response(str(output_list).encode('utf-8'), status=200,
                            mimetype='text/plain')
        response.headers['Content-Length'] = len(response.data)

    return response


if __name__ == '__main__':

    # We want to download our model before the server starts
    class GunicornServer(Command):

        description = 'Run the backend within Gunicorn'

        def __init__(self, host='127.0.0.1', port=5001, workers=6, timeout=3600,
                     worker_class="sync",
                     logger=None, download_model=False):
            self.port = port
            self.host = host
            self.workers = workers
            self.timeout = timeout
            self.worker_class = worker_class
            self.logger = logger
            self.download_model = download_model
            super().__init__()

        def get_options(self):
            return (
                Option('-h', '--host',
                       dest='host',
                       default=self.host),
                Option('-t', '--timeout',
                       dest='timeout',
                       default=self.timeout,
                       type=int),
                Option('-p', '--port',
                       dest='port',
                       type=int,
                       default=self.port),
                Option('-w', '--workers',
                       dest='workers',
                       type=int,
                       default=self.workers),
                Option('-k', '--worker-class',
                       dest='worker_class',
                       type=str,
                       default=self.worker_class),
                Option('-d', '--download-model',
                       dest="download_model",
                       type=bool,
                       default=self.download_model),
                Option('-l', '--logger',
                       dest="logger",
                       default=self.logger)
            )

        def __call__(self, application=None, *arguments, **kwargs):

            host = kwargs['host']
            port = kwargs['port']
            workers = kwargs['workers']
            worker_class = kwargs['worker_class']
            timeout = kwargs['timeout']
            download_model = kwargs['download_model']
            logger = kwargs['logger']

            # Download the model specified in the env. variable MODEL_TO_LOAD
            try:
                model_list = json.loads(os.environ['MODEL_TO_LOAD'])
            except (KeyError, JSONDecodeError):
                logger.error("Environmental variable MODEL_TO_LOAD not properly defined")
                return

            if len(model_list) != 2:
                logger.error("Env. Variable MODEL_TO_LOAD has the wrong format")
                return

            global model_name
            model_name = model_list[0]

            model_gdrive_id = model_list[1]

            if download_model:
                logger.info("Will donwload model before server starts")

                # download the model
                url = 'https://drive.google.com/uc?id=' + model_gdrive_id
                output = '/app/trained_models/' + model_name + ".zip"

                logger.debug(f"Downloading {url} to {output}")
                gdown.download(url, output, quiet=False)

                # uncompress the model
                logger.debug(f"Decompressing")
                with zipfile.ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join("/app/trained_models/", model_name))

                logger.info("Done. Starting WSGI server")

            logger.info("Started WSGI server")
            # clear kwargs
            self.server_options = {}

            class FlaskApplication(Application):
                def init(self, parser, opts, args):
                    return {
                        'bind': '{0}:{1}'.format(host, port),
                        'workers': workers,
                        'worker_class': worker_class,
                        'timeout': timeout
                    }

                def load(self):
                    return app

            FlaskApplication().run()

    manager = Manager(app)

    manager.add_command('gunicorn', GunicornServer(host='0.0.0.0',
                                                   port=5001,
                                                   workers=4,
                                                   worker_class="sync",
                                                   timeout=3600,
                                                   logger=app.logger,
                                                   download_model=True))

    parser = OptionParser()
    (options, args) = parser.parse_args()
    manager.run()
