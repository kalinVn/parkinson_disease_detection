import config
from service.ParkinsonDiseasePrediction import ParkinsonDiseasePrediction


class Service:

    def __init__(self):
        self.service_type = config.SERVICE_TYPE

    def get_service(self):
        if self.service_type == "ML":
            return ParkinsonDiseasePrediction()

