import numpy as np
import pandas as pd
import config

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from factory.Classifier import Classifier


class ParkinsonDiseasePrediction:

    def __init__(self):
        self.standard_data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        csv_path = config.CSV_PATH
        self.dataset = pd.read_csv(csv_path)

        self.scaler = StandardScaler()
        self.classifier_factory = Classifier()
        self.model = self.classifier_factory.get_model()

    def get_dataset(self):
        return self.dataset

    def preprocess(self):
        self.x = self.dataset.drop(columns=['name', 'status'], axis=1)
        self.y = self.dataset['status']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=2)
        self.standardize()

    def standardize(self):
        scaler = StandardScaler()
        scaler.fit(self.x_train )
        self.x_train = scaler.transform(self.x_train )
        self.x_test = scaler.transform(self.x_test)

    def build(self):
        self.model.fit(self.x, self.y)

    def test_accuracy_score(self):
        x_train_prediction = self.model.predict(self.x_train)
        training_data_accuracy = accuracy_score(self.y_train, x_train_prediction)
        print("Accuracy score of training data: ", training_data_accuracy)

        x_test_prediction = self.model.predict(self.x_test)
        test_data_accuracy = accuracy_score(self.y_test, x_test_prediction)
        print("Accuracy score of test data: ", test_data_accuracy)

    def predict(self, data):
        inputs = np.array([data])
        columns = list(self.dataset.columns)[:-1]

        # df = pd.DataFrame(inputs, columns=columns)

        prediction = self.model.predict(inputs)

        if prediction[0] == 1:
            print("The Person has Parkinson")
        else:
            print("he Person does not have Parkinson")

