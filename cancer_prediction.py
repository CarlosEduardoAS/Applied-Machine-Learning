import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

cancerdf = pd.DataFrame(data= np.c_[cancer['data'], cancer['target']],
                        columns= np.append(cancer['feature_names'], 
                        'target'))

X = cancerdf.drop('target', axis=1)
y = pd.Series(cancerdf['target'])

def fitknn():
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = KNeighborsClassifier(n_neighbors = 1)
    return knn.fit(X_train, y_train)

def prediction():
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = fitknn()
    return knn.predict(X_test)

def score():
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = fitknn()

    return knn.score(X_test, y_test)

def accuracy_plot():

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = fitknn()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y), 
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2), 
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)

accuracy_plot()
plt.show()

![Alt text](/PycharmProjects/Applied-Machine-Learning/Figure_1.png)
