from sklearn.metrics import confusion_matrix
import csv  

"""
Returrns CSV file with Accuracy, Precision, Recall, F_score, and Support for each tag and average

INPUT    test_name : used to create path for csv file
        test_words : must be list of tuples
    hidden_predict : return from inference method

OUTPUT  The function ccreates a csv file with the evaluation metrics seen above.
        Prints the average metrics for all the tags
        TODO: create path correctly

"""
def evaluate_predictions(test_name, test_words, hidden_predict):
    hidden_true = [w[1] for w in test_words]

    # seperate into seperate tag classes
    support_index = {}
    for i in range(len(hidden_true)):
        w = hidden_true[i]
        if w not in support_index.keys():
            support_index[w] = [i]
        else:
            support_index[w].append(i)

    header = ['Tag', 'Accuracy', 'Precision', 'Recall', 'F_score', 'Support']

    # calculate per tags and add to csv
    path = f'{test_name}.csv'
    with open(path, 'w+') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for sup in support_index:
            h_true = [hidden_true[i] for i in support_index[sup]]
            h_pred = [hidden_predict[i] for i in support_index[sup]]
            TN, FP, FN, TP = confusion_matrix(hidden_true, hidden_predict).ravel()
            accuracy = (TP+TN)/len(h_true)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f_score = 2 * ((Precision * Recall) / (Precision + Recall))

            data = [sup, accuracy, precision, recall, f_score, len(support_index[sup])]
            writer.writerow(data)

        # calculate for average and add to csv
        TN, FP, FN, TP = confusion_matrix(hidden_true, hidden_predict).ravel()
        Accuracy = (TP+TN)/len(h_true)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F_score = 2 * ((Precision * Recall) / (Precision + Recall))

        data = [test_name, Accuracy, Precision, Recall, F_score, len(hidden_true)]
        writer.writerow(data)

    print(f'The average result for experiment {test_name} are:')
    print(f'Accuracy: {data[1]}    Precision: {data[2]}    Recall: {data[2]}    F_score: {data[4]}')

    return data