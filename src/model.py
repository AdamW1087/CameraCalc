import numpy as np
import csv
import random
from numpy.random import default_rng
import math
import sys


# used by the tracker
def set_up_data(file_path):
    
    # collect data and labels from data file
    features, labels = load_data(file_path)
    
    # instantiate nn and load data
    nn = KNearestNeighbours()
    nn.train(features, labels)
    return nn



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


def compute_accuracy(actual_labels, predicted_labels, debug):
    
    # ensure label lists have the same length
    assert len(actual_labels) == len(predicted_labels)
    
    # count how many labels match
    count = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == predicted_labels[i]:
            count += 1
            
            
    # calculate and print debug info
    # "0" -> {[7, 1], [8, 11]}
    # "1" -> {[6, 6], [2, 1]}
    # ...
    # "9" -> {[1, 2], [4, 7]}
    if debug:
        # set up dictionary from what the label actually is
        incorrect_guesses = {}
        num_labels = np.arange(10).astype(str)
        op_labels = np.array(["+", "-", "*", "/"])
        for label in np.concatenate((num_labels, op_labels)):
            incorrect_guesses[label] = {}
            
            
        for i in range(len(actual_labels)):
            # filter for labels that arent matched
            if actual_labels[i] != predicted_labels[i]:
                # get map corresponding to the actual label
                guesses = incorrect_guesses[actual_labels[i]]
                
                # inc counter at guessed label by 1 or set to 1 if label has not been guessed yet
                if predicted_labels[i] in guesses:
                    inc_count = guesses[predicted_labels[i]]
                    inc_count += 1
                    guesses[predicted_labels[i]] = inc_count
                else:
                    guesses[predicted_labels[i]] = 1
                    
        # output all incorrect guesses
        for label in incorrect_guesses.keys():
            # get map for each label type
            attempts = incorrect_guesses[label]
            
            # print how many guesses each label had
            if len(attempts) != 0:
                # for the actual label
                print("For " + label + ":")
                
                # print incorrect guesses
                for fails in attempts.items():
                    print(fails[0] + " was guessed " + str(fails[1]) + " times")
                
                # separate output for easier reading
                print()
            
    # calculate decimal of how many are correct
    return float(count / len(actual_labels))


class KNearestNeighbours:
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

def test_model(file_path, debug):
    
    # load from known data
    features, labels = load_data(file_path)

    # split the data into training and testing sets (80% train, 20% test)
    seed = 674920
    rg = default_rng(seed)
    
    coords_train, coords_test, labels_train, labels_test = split_dataset(features, labels,
                                                 test_proportion=0.2,
                                                 random_generator=rg)
    
    
    # create nn model
    nn = KNearestNeighbours()
    
    # train model using training data
    nn.train(coords_train, labels_train)
    
    # predict labels for test data
    labels_predictions = nn.predictMultiple(coords_test, 1)
    
    # calculate accuracy of model
    accuracy = compute_accuracy(labels_test, labels_predictions, debug)
    
    print(accuracy)

def main():
    file_path = 'src/point_history.csv'
    # setup for modes
    debug = False
    
    # check user input for mode choice
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        match mode:
            case "debug":
                debug = True
                
                
    # tests accuracy of model
    test_model(file_path, debug)

if __name__ == '__main__':
    main()
