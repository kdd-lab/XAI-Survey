import sys

from exputil import get_autoencoder
from exputil import get_dataset
import shap
import warnings
warnings.filterwarnings('ignore')


def main():

    #select the dataset: cifar10, net or mnist
    dataset = 'cifar10'
    ae_name = 'aae'
    #define the epochs and the batch size
    epochs = 40
    batch_size = 256
    sample_interval = 200
    #get the dataset and train the autoencoder
    path = '../'
    path_aemodels = path + 'aemodels/%s/%s/' % (dataset, ae_name)
    _, _, X_test, Y_test, use_rgb = get_dataset(dataset)
    print(X_test.shape)
    print(Y_test.shape)
    ae = get_autoencoder(X_test, ae_name, dataset, path_aemodels)
    print(ae)
    ae.fit(X_test, epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
    ae.save_model()
    ae.sample_images(epochs)


if __name__ == "__main__":
    main()
