from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# draw graph in order to show the best value of k
def draw_graph(krange_, score):
    plt.scatter(k_range, score)
    plt.xlabel('value of K')
    plt.ylabel('Testing Accuracy')
    plt.show()

# apply knn model to train data, and predict the test data.
# this method, then, compares the predict data with test response data, and measure the accuracy
def apply_KNN_model(k):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_predict = knn_classifier.predict(X_test)
    score_accuracy = metrics.accuracy_score(y_test, y_predict)
    return score_accuracy


# this method train the knn with entire data set, and predict unknown sample data sets
def apply_KNN_predict(k):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X,y)
    sample_list = [[1,2,3,4],[3,5,2,1]]
    print(knn_classifier.predict(sample_list))


# load iris data set
data_iris = load_iris()

# X is a matrix which contains the attributes or features of the data set
X = data_iris.data

# y is a vector which contains the response of the data set
y = data_iris.target

# split the data set into two parts: training data set (70%) and testing data set (30% of the entire data)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

# print(X_train.shape)
# print(y_train.shape)


k_range = range(1,15)
score = []
for k in range(1,15):
    acc_score = apply_KNN_model(k)
    score.append(acc_score)


# print(score)
# print(max(score))

# find the index of the maximum value from the score list
max_k = score.index(max(score))

# print(max_k)

draw_graph(k_range, score)

# apply knn with the best value of K
apply_KNN_predict(max_k)
