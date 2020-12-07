#### Import Libraries --------------------------------------------------------------------------------------------------
from random import seed, randint
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


#import models
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#evaluate modules
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score

#### Define functions---------------------------------------------------------------------------------------------------
def two_class_confusion_matrix(i):
    # make a confussion matrix
    cm = confusion_matrix(y_test, y_predicted[i])

    # make the variables
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    c_matrix = pd.DataFrame([[TN, FP], [FN, TP]], columns=["P: Negative", "P: Positive"],
                            index=["A: Negative", "A: Positive"])

    print(f"the confussion matrix of the {model_names[i]}\n")
    print(c_matrix)
    print('\nAccuracy', round((TN + TP) / len(y_predicted[i]) * 100, 1), "%")
    return True


def three_class_confusion_matrix(i):
    cm = confusion_matrix(y_test, y_predicted[i])
    TP_0 = cm[0][0]
    TP_1 = cm[1][1]
    TP_2 = cm[2][2]
    TN_0 = TP_1 + TP_2
    TN_1 = TP_0 + TP_2
    TN_2 = TP_0 + TP_1
    FP_0 = cm[[1, 2], 0]  # select the FP of cluster 1 and 2
    FP_1 = cm[[0, 2], 1]  # select the FP of cluster 0 and 2
    FP_2 = cm[[0, 1], 2]
    FN_0 = cm[0, [1, 2]]
    FN_1 = cm[1, [0, 2]]
    FN_2 = cm[2, [0, 1]]

    # put all values into the confusion matrix
    values = [[TP_0, TN_0, FP_0, FN_0],
              [TP_1, TN_1, FP_1, FN_1],
              [TP_2, TN_2, FP_2, FN_2]]

    # put confusion matrix into dataframe
    c_matrix = pd.DataFrame(values, columns=["TP", "TN", "FP", "FN"],
                            index=["Class 0", "Class 1", "Class 2"])

    print(f"the confusion matrix\n",
          "left is always low to high: for class one FP class [one, two]\n",
          c_matrix)
    accuracy = metrics.accuracy_score(y_test, y_predicted[i])
    print(f"The accuracy of the model: {round(accuracy * 100, 1)} %")

    return True

#### Import data -------------------------------------------------------------------------------------------------------
# import data
data = pd.read_csv("heart_failure_clinical_records_dataset.csv")
# declare the variables
#data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
x = data.iloc[:, 0:11].values
y = data.iloc[:, 12].values


# Split data into training and test sets
seed(9999) # set seed for reproducing results
pseudo_random_number = randint(1, 100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state= pseudo_random_number)

# apply feature scaling --> you normalize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#### Models -------------------------------------------------------------------------------------------------------
# import the support vector machine
# all versions: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
C = 1000
svm_linear = SVC(kernel= 'linear', random_state=pseudo_random_number, C=C)
svm_poly = SVC(kernel= 'poly', random_state=pseudo_random_number, C=C)
svm_rbf = SVC(kernel= 'rbf', random_state=pseudo_random_number, C=C)
svm_sigmoid = SVC(kernel= 'sigmoid', random_state=pseudo_random_number, C=C)

# import the naieve bayers
naive_bayes = GaussianNB()

# import the logsitic regression
log_reg = LogisticRegression(random_state=pseudo_random_number)

# train the models using the training set
svm_linear.fit(x_train, y_train)
svm_poly.fit(x_train, y_train)
svm_rbf.fit(x_train, y_train)
svm_sigmoid.fit(x_train, y_train)

naive_bayes.fit(x_train, y_train)

log_reg.fit(x_train, y_train)

#### evaluation of the model -------------------------------------------------------------------------------------------
# predict the test set, and put the result in a list
y_predicted = []

y_predicted.append(svm_linear.predict(x_test))
y_predicted.append(svm_poly.predict(x_test))
y_predicted.append(svm_rbf.predict(x_test))
y_predicted.append(svm_sigmoid.predict(x_test))

y_predicted.append(naive_bayes.predict(x_test))

y_predicted.append(log_reg.predict(x_test))

#### evaluation of the model -------------------------------------------------------------------------------------------
model_names = ["svm_linear", "svm_poly", "svm_rbf", "svm_sigmoid",  "naive_bayes", "log_reg"]

for i, j in enumerate(y_predicted):
    print("-*" * 50)
    print(f"performance of model {model_names[i]}\n")
    print(classification_report(y_test, y_predicted[i]))

    unique_elements = np.unique(y)

    if len(unique_elements) == 2:
        print_confusion_matrix = two_class_confusion_matrix(i)
    elif len(unique_elements) == 3:
        print_confusion_matrix = three_class_confusion_matrix(i)
    else:
        raise Exception("y not discrete or larger than four")

    print("-*" * 50)
    print("\n\n\n")
