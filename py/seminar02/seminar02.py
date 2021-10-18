import numpy as np
import csv
import os
import sys

if __name__ == '__main__':
    # Load data
    track, x, y = [], [], []
    with open(os.path.join(sys.path[0], "input/125.csv"), newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            track.append(row[1])
            x.append(float(row[3]))
            y.append(float(row[4]))

    # Get unique objects and map them to their trajectories
    track_ = set(track)
    track_ = dict(zip(track_, range(len(track_))))
    print('Objects num:', len(track_))

    # Setup trajectories
    X, Y = [], []
    for i in range(len(track_)):
        X.append([])
        Y.append([])

    # Extract trajectories
    for i in range(len(track)):
        X[track_[track[i]]].append(x[i])
        Y[track_[track[i]]].append(y[i])

    # Convert to numpy for further evaluation
    for i in range(len(track_)):
        X[i] = np.array(X[i])
        Y[i] = np.array(Y[i])

    # Compute lengths of the trajectories
    lengths = []
    for i in range(len(track_)):
        lengths.append(np.sum(np.sqrt(np.diff(X[i]) ** 2 + np.diff(Y[i]) ** 2)))

    lengths = np.array(lengths)

    # Get max length
    print('Max length: %.2f' % np.max(lengths))