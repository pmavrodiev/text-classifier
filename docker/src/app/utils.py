from datetime import datetime
from bs4 import BeautifulSoup


class Document:
    """
    A utility class that represents a document.
    We use the same Document representation for "training" and "test" documents

    The difference between a training and test document is mainly in the available fields:
        1. A training Document has 'labels', whereas a test Document does not
        2. The content of a training or test document is assumed to be a dictionary as specified in the
           json schema of train_data.json and predict_data.json.
           However if the content is a simple string, we assume that the user has invoked /predict_raw and has thus
           supplied us with the raw text on which to do prediction
    """

    @staticmethod
    def get_datetime(date_str):
        """
        A utility function to convert a datetime string into a datetime object
        """
        date_str_cleaned = date_str.split('+')
        if len(date_str_cleaned) == 1:
            try:
                timestamp = datetime.strptime(date_str_cleaned[0], '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                timestamp = datetime.strptime(date_str_cleaned[0], '%Y-%m-%dT%H:%M:%S')
        elif len(date_str_cleaned) == 2:
            try:
                assert date_str_cleaned[1] == "00:00"
                timestamp = date_str_cleaned[0]
            except AssertionError:
                print(date_str)

        return timestamp

    def __init__(self, metadata=None, content="", labels={}):
        self.title = ""
        # does the Document contain raw text or a payload as given in the json schema of the challenge
        self.raw_text = True  # by default assume it's raw text
        if metadata is not None:
            # this is a text following the schema of the challenge
            # convert to datetime object
            self.publishedAt = Document.get_datetime(metadata['publishedAt'])
        if isinstance(content, dict):
            # a train or test document following the json schema of the challenge
            self.raw_text = False
            if "title" in content:
                self.title = content['title'].replace("\n", "").replace("\t", "")
            if "fullTextHtml" in content:
                # we strip the html tags and remove a few annoying characters
                self.content = ''.join(
                    BeautifulSoup(content['fullTextHtml'], "html.parser").text.replace("\n", "").replace("\t", ""))

            self.sections = []
            if "sections" in content:
                # only train documents has Sections
                for sec in content['sections']:
                    self.sections.append(sec)
        elif isinstance(content, str):
            # this is a raw text supplied to /predict_raw
            self.content = content.replace("\n", "").replace("\t", "")
        else:
            self.content = ""

        self.labels = {}
        self.has_labels = False
        for label in labels:
            # replace ',' by '-', because Farm expects ',' as a separator in multilabel classification
            # for final replacing we'll replace it back
            self.labels[label[0].replace(",", "-")] = label[1]  # taxonomy name -> confidence
            self.has_labels = True

    def get_text(self):
        """
        Get the raw text of a Document. This is simply a concatenation of the Title and the Content.
        """
        if self.raw_text:
            return self.content
        else:
            if self.title != "":
                # add the title as the first sentence in the text
                return self.title + "." + self.content
            return self.content

    def get_training_sample(self):
        """
        Create a training sample from this Document.
        The format is:
          <document text><tab><comma-separated list of training labels>
        :return:
        """
        if self.has_labels:
            labels = ','.join(list(self.labels.keys()))
            yield self.get_text() + "\t" + labels
        else:
            return None
