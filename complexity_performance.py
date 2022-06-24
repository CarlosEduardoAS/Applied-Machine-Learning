import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#Regression

np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


def plot_data():
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4)
    plt.show()


def polynomial_regression():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    
    result = np.zeros((4, 100))
    

    for i, degree in enumerate([1, 3, 6, 9]):
        poly = PolynomialFeatures(degree = degree)
        X_poly = poly.fit_transform(X_train.reshape(11, 1))
        linreg = LinearRegression().fit(X_poly, y_train)
        y = linreg.predict(poly.fit_transform(np.linspace(0, 10, 100).reshape(100, 1)));
        result[i,:] = y
    return result 


def poly_r2_scores():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    r2_train = np.zeros(10)
    r2_test = np.zeros(10)
    
    for i in range(10):
        poly = PolynomialFeatures(degree=i)
        
        # Train and score x_train
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_poly, y_train)        
        r2_train[i] = linreg.score(X_poly, y_train)
        
        # Score x_test (do not train)
        X_test_poly = poly.fit_transform(X_test.reshape(4,1))
        r2_test[i] = linreg.score(X_test_poly, y_test)
        
    return (r2_train, r2_test)


def score_analysis():
    
    r2_scores = poly_r2_scores()
    df = pd.DataFrame({'training_score':r2_scores[0], 'test_score':r2_scores[1]})
    df['diff'] = df['training_score'] - df['test_score']
    
    df = df.sort(['diff'])
    good_gen = df.index[0]
    
    df = df.sort(['diff'], ascending = False)
    overfitting = df.index[0]
    
    df = df.sort(['training_score'])
    underfitting = df.index[0]
    
    return (underfitting, overfitting, good_gen)


def linear_lasso_r2_score():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Create Polinomial Features
    poly = PolynomialFeatures(degree=12)
    
    # Shape Polinomial Features
    X_train_poly = poly.fit_transform(X_train.reshape(11,1))
    X_test_poly = poly.fit_transform(X_test.reshape(4,1))
    
    # Linear Regression
    linreg = LinearRegression().fit(X_train_poly, y_train)
    lin_r2_test = linreg.score(X_test_poly, y_test)

    # Lasso Regression
    linlasso = Lasso(alpha=0.01, max_iter = 10000).fit(X_train_poly, y_train)
    las_r2_test = linlasso.score(X_test_poly, y_test)
    
    return (lin_r2_test, las_r2_test) 


#Classification

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

X_subset = X_test2
y_subset = y_test2


def most_important_features():
    from sklearn.tree import DecisionTreeClassifier
    
    tree_clf = DecisionTreeClassifier().fit(X_train2, y_train2)
    
    feature_names = []
    
    for index, importance in enumerate(tree_clf.feature_importances_):
        feature_names.append([importance, X_train2.columns[index]])
    
    feature_names.sort(reverse=True)
    feature_names = np.array(feature_names)
    feature_names = feature_names[:5,1]
    feature_names = feature_names.tolist()
    
    return feature_names


def models_mean_scores():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    svc = SVC(kernel='rbf', C=1, random_state=0)
    gamma = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(svc, X_subset, y_subset,
                            param_name='gamma',
                            param_range=gamma,
                            scoring='accuracy')

    scores = (train_scores.mean(axis=1), test_scores.mean(axis=1))
        
    return scores 


