import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # index each month to its particular value
    month_assign = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4,
                    'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9,
                    'Nov': 10, 'Dec': 11}

    # Links appropriate value to visitors and labels
    visitor_weekend = {'Returning_Visitor': 1, 'New_Visitor': 0, 'Other': 0, 'TRUE': 1, 'FALSE': 0}

    # assigns data types (int or float) according to description
    value_assign = {0: (lambda x: (int(x))), 1: (lambda x: (float(x))),
                    2: (lambda x: (int(x))), 3: (lambda x: (float(x))),
                    4: (lambda x: (int(x))), 5: (lambda x: (float(x))),
                    6: (lambda x: (float(x))), 7: (lambda x: (float(x))),
                    8: (lambda x: (float(x))), 9: (lambda x: (float(x))),
                    10: (lambda x: (month_assign[x])), 11: (lambda x: (int(x))),
                    12: (lambda x: (int(x))), 13: (lambda x: (int(x))),
                    14: (lambda x: (int(x))), 15: (lambda x: (visitor_weekend[x])),
                    16: (lambda x: (visitor_weekend[x]))}

    with open(filename) as file_name:
        contents = csv.reader(file_name)
        # skips the first line (column headers)
        next(contents)
        evidence = []
        labels = []
        for line, row in enumerate(contents):
            temp_evidence = []
            for index, value in enumerate(row):
                # last index is put in labels
                if index == 17:
                    labels.append(visitor_weekend[value])
                else:
                    temp_evidence.append(value_assign[index](value))
            evidence.append(temp_evidence)

        return (evidence, labels)



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors = 1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive_total, positive_rate = 0, 0
    negative_total, negative_rate = 0, 0
    for i in range(len(labels)):
        
        # sensitivity
        if labels[i] == 1:
            positive_total += 1
            if labels[i] == predictions[i]:
                positive_rate += 1
        
        # specificity
        else:
            negative_total += 1
            if labels[i] == predictions[i]:
                negative_rate += 1

    sensitivity = float(positive_rate / positive_total)
    specificity = float(negative_rate / negative_total)
    
    return (sensitivity, specificity)


if __name__ == "__main__":
    main()
