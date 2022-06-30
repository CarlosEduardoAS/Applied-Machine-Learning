import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def fraud_percentage():
    return df['Class'].sum()/len(df['Class'])


def dummy_scores():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    
    dummy_clf = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    y_dummy_predictions = dummy_clf.predict(X_test)
    
    return (dummy_clf.score(X_test, y_test), recall_score(y_test, y_dummy_predictions))


def SVC_scores():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    svm = SVC().fit(X_train, y_train)
    svm_predictions = svm.predict(X_test)
    
    return (svm.score(X_test, y_test), recall_score(y_test, svm_predictions), precision_score(y_test, svm_predictions))


def SVC_confusion_matrix():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    svm = SVC(C=1e9, gamma=1e-07).fit(X_train, y_train)
    scores = svm.decision_function(X_test)
    
    y_pred_with_threshold = scores > -220
    
    return confusion_matrix(y_test, y_pred_with_threshold)


def draw_pr_curve():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve, auc

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_scores_lr = lr.decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()

#draw_pr_curve()


def draw_roc_curve():
    from sklearn.linear_model import LogisticRegression
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_scores_lr = lr.decision_function(X_test)
    
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    
#draw_roc_curve()

def lr_GridSearch():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    grid_values = {'penalty': ['l1', 'l2'], 'C':[0.01, 0.1, 1, 10, 100]}
    
    grid = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
    grid.fit(X_train, y_train)
    
    return grid.cv_results_['mean_test_score'].reshape(-1, 2)


def GridSearch_Heatmap(scores):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0)

#GridSearch_Heatmap(lr_GridSearch())