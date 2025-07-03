#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 31st December 2017 - new year eve :)
# ----------------------------------------------

import csv
import random
import math
import operator
import cv2


# calculation of euclidead distance
def calculateEuclideanDistance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)


# get k nearest neigbors
def kNearestNeighbors(training_feature_vector, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(training_feature_vector)):
        dist = calculateEuclideanDistance(testInstance,
                training_feature_vector[x], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# votes of neighbors
def responseOfNeighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sortedVotes = sorted(all_possible_neighbors.items(),
                         key=operator.itemgetter(1), reverse=True)
    
    # Return the predicted class and the count of votes for that class
    return sortedVotes[0][0], sortedVotes[0][1]


# Load image feature data to training feature vectors and test feature vector
def loadDataset(
    filename,
    filename2,
    training_feature_vector=[],
    test_feature_vector=[],
    ):
    # Clear existing data to prevent accumulation on multiple calls
    training_feature_vector.clear()
    test_feature_vector.clear()

    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            # Ensure data has at least 4 columns (R, G, B, Label)
            if len(dataset[x]) >= 4:
                try:
                    for y in range(3): # Convert R, G, B to float
                        dataset[x][y] = float(dataset[x][y])
                    training_feature_vector.append(dataset[x])
                except ValueError:
                    print(f"Skipping malformed training data row: {dataset[x]}")
            else:
                print(f"Skipping incomplete training data row: {dataset[x]}")


    with open(filename2) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            # Ensure data has at least 3 columns (R, G, B)
            if len(dataset[x]) >= 3:
                try:
                    for y in range(3): # Convert R, G, B to float
                        dataset[x][y] = float(dataset[x][y])
                    test_feature_vector.append(dataset[x])
                except ValueError:
                    print(f"Skipping malformed test data row: {dataset[x]}")
            else:
                print(f"Skipping incomplete test data row: {dataset[x]}")


def main(training_data, test_data, k_value=3):
    training_feature_vector = []  # training feature vector
    test_feature_vector = []  # test feature vector
    loadDataset(training_data, test_data, training_feature_vector, test_feature_vector)
    
    if not test_feature_vector:
        return "No test data", 0 # Return a default if no test data is loaded

    classifier_prediction = []  # predictions
    k = k_value  # K value of k nearest neighbor
    
    # Ensure k is not greater than the number of training samples
    if k > len(training_feature_vector):
        k = len(training_feature_vector) if len(training_feature_vector) > 0 else 1
        print(f"Warning: K value adjusted to {k} as it was greater than training data size.")

    for x in range(len(test_feature_vector)):
        neighbors = kNearestNeighbors(training_feature_vector, test_feature_vector[x], k)
        result, votes = responseOfNeighbors(neighbors)
        classifier_prediction.append((result, votes, k)) # Store prediction, votes, and k
    
    # For a single test instance, return its prediction, votes, and k
    if classifier_prediction:
        return classifier_prediction[0][0], classifier_prediction[0][1], classifier_prediction[0][2]
    else:
        return "N/A", 0, k # Fallback if no prediction was made

