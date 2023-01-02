from factory.Service import Service
from visualizator import dist_plot, countplot


def parkinson_disease_prediction():
    factory_service = Service()
    service = factory_service.get_service()
    # print(service)

    service.preprocess()

    service.build()
    service.test_accuracy_score()

    data = [116.01400, 141.78100, 110.65500, 0.01284, 0.00011, 0.00655, 0.00908, 0.01966, 0.06425, 0.58400, 0.03490,
            0.04825, 0.04465, 0.10470, 0.01767, 19.64900, 0.417356, 0.823484, -3.747787, 0.234513, 2.332180, 0.410335]
    service.predict(data)

    df = service.get_dataset()
    # title = 'MDVP:Fhi(Hz) Distribution'
    # dist_plot(df['MDVP:Fhi(Hz)'], title)

    # params = dict(x='status', data=df)
    # countplot(df, params)


parkinson_disease_prediction()






