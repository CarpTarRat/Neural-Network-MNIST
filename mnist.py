import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy as np

class mnistOpener:
    def __init__(self, num_images: int, mini_batch: int, validation: int, test: int):
        self.num_images = num_images
        self.mini_batch = mini_batch
        self.validation = validation
        self.test = test

    def open_files(self):
        training_file = open("train-images.idx3-ubyte", 'rb') #60000 training datarb binary format
        training_labels_file = open('train-labels.idx1-ubyte', 'rb')

        training_file.read(16) #first 16 bytes skip
        training_buf = training_file.read(28 * 28 * self.num_images) #read remaining bytes, 28x28= image size
        training_labels_file.read(8)
        labels_buf = training_labels_file.read(self.num_images)

        data = np.frombuffer(training_buf, dtype=np.uint8).astype(np.float32) / 255 #254 scaling to[0,1] no normalisation to mean 0, sd =1 yet
        data_image = data.reshape(self.num_images // self.mini_batch, self.mini_batch, 28, 28) #groups into mini_batch, this is for viewing image
        data = data.reshape(self.num_images // self.mini_batch, self.mini_batch, 784)
        labels_data = np.frombuffer(labels_buf, dtype=np.uint8)
        labels_data = np.array([np.array([1 if i == number else 0 for i in range(10)]) for number in labels_data])
        labels_data = labels_data.reshape(self.num_images // self.mini_batch, self.mini_batch, 10)
        return (data, labels_data, data_image)
    
    def split_data(self):
        training_number = self.num_images // self.mini_batch - self.validation - self.test
        data = self.open_files()
        training_data = np.array([data[0][i] for i in range(training_number)])
        training_labels = np.array([data[1][i] for i in range(training_number)])
        validation_data =  np.array([data[0][i] for i in range(training_number, training_number + self.validation )])
        validaton_labels = np.array([data[1][i] for i in range(training_number, training_number + self.validation )])
        testing_data = np.array([data[0][i] for i in range(training_number + self.validation, training_number + self.validation + self.test)])
        testing_labels = np.array([data[1][i] for i in range(training_number + self.validation, training_number + self.validation + self.test)])
        return training_data, validation_data, testing_data, training_labels, validaton_labels, testing_labels, data[2]

def show_image(file):
    image = np.asarray(file).squeeze()
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    #np.set_printoptions(threshold=sys.maxsize)
    test = mnistOpener(50000, 10, 0, 1)
    files = test.split_data()
    #print(files[5])
    #np.set_printoptions(threshold = False)
    image = np.asarray(files[6][4999][9]).squeeze()
    #print(image)
    plt.imshow(image)
    plt.show()
    