import csv

from nndiy.linalg import Vector, Matrix2D


def read_data(path, label_column=0, sep=',', skip_lines=1, nrows=-1):
    rows = []
    labels = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=sep)
        counter = 0
        for row in reader:
            if skip_lines > 0:
                skip_lines -= 1
                continue
            if (nrows > 0) and (counter == nrows):
                break
            row = [float(i) for i in row]
            labels.append(row.pop(label_column))
            rows.append(Vector(row))
            counter += 1

    return Matrix2D(rows).T(), Vector(labels)
