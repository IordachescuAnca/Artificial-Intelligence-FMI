import numpy as np
import sys

train_images = np.load('data/train_images.npy')
train_labels = np.load('data/train_labels.npy')

class Knn_classifier:
    def __init__(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
    
    def classify_image(
        self, test_image,
        num_neighbors = 3, metric = 'l2'
    ):
        distances = np.sqrt(np.sum(
            np.square(self.train_images - test_image),
            axis = 1
        ))
        indexes = np.argsort(distances)
        indexes = indexes[:num_neighbors]
        labels = self.train_labels[indexes]
        label = np.argmax(np.bincount(labels))

        return label



test_images = np.load('data/test_images.npy')
test_labels = np.load('data/test_labels.npy')

knn = Knn_classifier(train_images, train_labels)
print(
    knn.classify_image(test_images[0])
)


