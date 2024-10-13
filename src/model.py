import numpy as np
import csv
import random
from numpy.random import default_rng
import math


# used by the tracker
def set_up_data(file_path):
    
    # collect data and labels from data file
    features, labels = load_data(file_path)
    
    # instantiate nn and load data
    nn_classifier = NearestNeighbour()
    nn_classifier.train(features, labels)
    return nn_classifier



def load_data(csv_path):
    # create lists to split the coordinates and corresponding labels
    coords = []
    labels = []
    
    with open(csv_path, 'r') as file:
        
        # read file one data entry at a time
        reader = csv.reader(file)
        for row in reader:
            
            # extract label
            label = row.pop(0)
            
            # adds [(x1,y1), (x2,y2)...] to coords
            coords.append(format_data(row))
            
            # add label at same index
            labels.append(label)

    return (coords, labels)


def format_data(row):
    coords = []
    
    for i in range(len(row)):
        
        # remove csv string padding
        row[i] = row[i].strip("[]").split(", ")
        
        # convert entries to floats and format as a pair
        coords.append((float(row[i][0]), float(row[i][1])))
        
    return coords



def split_dataset(coords, labels, test_proportion, random_generator=random):
    # generate shuffled indices
    indices = list(range(len(coords)))
    random_generator.shuffle(indices)

    # shuffle both coords and labels using the same indices
    coords_shuffled = [coords[i] for i in indices]
    labels_shuffled = [labels[i] for i in indices]

    # calculate the split index
    test_size = int(test_proportion * len(coords))

    # split the dataset into training and test sets
    coords_train, coords_test = coords_shuffled[test_size:], coords_shuffled[:test_size]
    labels_train, labels_test = labels_shuffled[test_size:], labels_shuffled[:test_size]
    
    return coords_train, coords_test, labels_train, labels_test


def compute_accuracy(actual_labels, predicted_labels):
    
    # ensure label lists have the same length
    assert len(actual_labels) == len(predicted_labels)
    
    # count how many labels match
    count = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == predicted_labels[i]:
            count += 1
            
    # calculate decimal of how many are correct
    return float(count / len(actual_labels))


class NearestNeighbour:
    # store known coords and labels
    def __init__(self):
        self.coords = np.array([])
        self.labels = np.array([])

    # save training data
    def train(self, coords, labels):
        self.coords = coords
        self.labels = labels

    # predicts a list of data entries (testing the model)
    def predictMultiple(self, coords, k):
        labels = []
        for i in range(len(coords)): 
            labels.append(self.predictSingle(coords[i], k))
        return np.array(labels)
    
    # predicts a single data entry (live guesses)
    def predictSingle(self, coords, k):
        dists = []
        
        # check all known values to see which is the closest to current coordinate
        for i in range(len(self.coords)): 
            sum = 0
            
            for j in range(min(len(coords), len(self.coords[i]))):
                # computes dist between each pair of coords
                sum += math.sqrt(((coords[j][0] - self.coords[i][j][0]) ** 2) + ((coords[j][1] - self.coords[i][j][1]) ** 2))
            # get avg distance
            sum /= float(min(len(coords), len(self.coords[i])))
            
            # use binary search to input new distance
            dists = input_sum(dists, (sum, i))

                
        # return nearest neighbour if k is 1
        if k == 1:
            return self.labels[dists[0][1]]
        
        # initialise neighbours dictionary ({label, count of label})
        neighbours = {}
        
        for i in range(len(dists)):
            # if the label is in neighbours, inc its count
            if self.labels[dists[i][1]] in neighbours:
                neighbours[self.labels[dists[i][1]]] += 1
                
                # if the count is k, return label
                if int(neighbours[self.labels[dists[i][1]]]) == k:
                    return self.labels[dists[i][1]]
            else:
                # new label to neighbours, set count to 1
                neighbours[self.labels[dists[i][1]]] = 1


def input_sum(dists, dist_index):
    # input pair based on dist
    dist, _ = dist_index
    
    # intialise the 2 boundaries
    start = 0
    end = len(dists) - 1
    
    while start <= end:
        mid = (start + end) // 2
        
        # shorten window to find desired index
        if dists[mid][0] < dist:
            start = mid + 1 
        else:
            end = mid - 1 
        
    # input (dist, index) into dists
    dists.insert(start, dist_index)
    
    return dists

def test_model(file_path):
    
    # load from known data
    features, labels = load_data(file_path)

    # split the data into training and testing sets (80% train, 20% test)
    seed = 398264
    rg = default_rng(seed)
    
    coords_train, coords_test, labels_train, labels_test = split_dataset(features, labels,
                                                 test_proportion=0.2,
                                                 random_generator=rg)
    
    
    # create nn model
    nn_classifier = NearestNeighbour()
    
    # train model using training data
    nn_classifier.train(coords_train, labels_train)
    
    # predict labels for test data
    labels_predictions = nn_classifier.predictMultiple(coords_test, 1)
    
    # calculate accuracy of model
    accuracy = compute_accuracy(labels_test, labels_predictions)
    
    print(accuracy)

def main():
    file_path = 'src/point_history.csv'

    # tests accuracy of model
    test_model(file_path)

if __name__ == '__main__':
    main()
