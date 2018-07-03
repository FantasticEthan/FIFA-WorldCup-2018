import sys
sys.path.append('../')
"""
Created on Tue June 26 12:09:29 2018

@author: ethan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.labelEncoder import loadLabelEncoder,saveLabelEncoder
from utils.result_report import plot_confusion_matrix,plot_ROC_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


class candidateModel():
    """candidata model for predict and fit """
    """ author：ethan """

    def __init__(self,ModelName,GridSearchCV=True,Confusion=True,ROC=True,fold_test=10,ifsave=True):
        """
        Initializes the candidateModel.
        Args:
          ModelName : modelname:{LR,SVM,RF}.
          GridSearchCV : if gridSearch or not
          Confusion : plot confusion matrix
          ROC : plot ROC image
          fold_test : k fold test
          ifsave : save the model.
        """
        # self.params = params
        self.modelname = ModelName
        self.grid_search = GridSearchCV
        self.confusion = Confusion
        self.roc = ROC
        self.fold_test = fold_test
        self.save = ifsave
        self.le_result = loadLabelEncoder('../../data/LE/result.npy')
        self.class_names = self.le_result.classes_
        # self.model = modelDict[self.modelname]


    def fit(self, X, y):
        """ fit model. """
        if self.modelname=='LR':
            model_LR = LogisticRegression(multi_class='multinomial',
                                          solver = 'newton-cg',
                                          ).fit(X, y)
            train_accuracy = model_LR.score(X, y)
            print('train accuracy is {}'.format(train_accuracy))
            self.model = model_LR

        if self.modelname=='RF':
            model_RF = RandomForestClassifier(n_estimators=100,
                                              oob_score=True).fit(X, y)
            train_accuracy = model_RF.score(X, y)
            print('train accuracy is {}'.format(train_accuracy))
            print('oob_score_ is {}'.format(model_RF.oob_score_))
            self.model = model_RF


    def eval(self,X_test,y_test):
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))
        if self.confusion == True:
            cnf_matrix = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cnf_matrix, classes=self.class_names,
                                  title=str(self.modelname) +
                                        ' Confusion matrix, without normalization')
        if self.roc == True:
            if self.modelname!='RF':
                y_score = self.model.decision_function(X_test)
                plot_ROC_curve(y_test, y_score, title=str(self.modelname) +' ROC curve',
                               class_names=self.class_names)

        plt.close()

    def predict(self, X_pred):
        """ predict test data. """
        result = pd.DataFrame()
        if self.modelname =='LR':
            y_pre = self.model.predict(X_pred)



if __name__ == '__main__':
    df = pd.read_csv('../../result/process_data/train.csv',)
    df_X = df.iloc[:,:-2]
    df_y = df.ix[:,['result']]
    X_train,X_test,y_train,y_test = train_test_split(df_X,df_y,
                                                     test_size=0.2,
                                                     random_state=42)
    model = candidateModel(ModelName='RF')
    model.fit(X_train,y_train)
    model.eval(X_test,y_test)
