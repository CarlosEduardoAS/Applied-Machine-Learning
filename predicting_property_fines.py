import pandas as pd
import numpy as np


def blight_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    
    df = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    df.index = df['ticket_id']
    features_name = ['fine_amount', 'admin_fee', 'state_fee', 'late_fee']
    df.compliance = df.compliance.fillna(value=-1)
    df = df[df.compliance != -1]
    
    df_test = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    df_test.index = df_test['ticket_id']

    X = df[features_name]
    X.fillna(value = -1)
    y = df.compliance
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    clf = RandomForestClassifier(n_estimators = 10, max_depth = 5).fit(X_train, y_train)

    X_predict = clf.predict_proba(df_test[features_name])
    ans = pd.Series(data = X_predict[:,1], index = df_test['ticket_id'], dtype='float32')
            
    return ans


blight_model()