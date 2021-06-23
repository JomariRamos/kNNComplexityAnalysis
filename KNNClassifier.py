import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
import math

class KNNClassifier() : 
      
    def __init__(self, K):
        self.elementary_operation_counter = 0
        self.K = K
          
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
          
        # no_of_training_examples, no_of_features
        self.m, self.n = X_train.shape
      
    def predict(self, X_test):
        self.X_test = X_test
          
        # no_of_test_examples, no_of_features
        self.m_test, self.n = X_test.shape
          
        Y_predict = np.zeros(self.m_test)
          
        for i in range(self.m_test):
            x = self.X_test[i]
              
            # find the K nearest neighbors from current test example
            neighbors = np.zeros(self.K)
            neighbors = self.find_neighbors(x)
            
            # most frequent class in K neighbors
            Y_predict[i] = mode(neighbors)[0][0]    
              
        return Y_predict, self.elementary_operation_counter
      
    def find_neighbors(self, x):
          
        # calculate all the euclidean distances between current 
        # test example x and training set X_train

        euclidean_distances = np.zeros(self.m)
        for i in range(self.m): 

            #Calculating the operation counter
            self.elementary_operation_counter += 1 

            d = self.euclidean(x, self.X_train[i])
            euclidean_distances[i] = d
          
        # sort Y_train according to euclidean_distance_array and 
        # store into Y_train_sorted
          
        inds = euclidean_distances.argsort()
    
        Y_train_sorted = self.Y_train[inds]
          
        return Y_train_sorted[:self.K]
     
    def euclidean(self, x, x_train) :
        return np.sqrt(np.sum(np.square(x - x_train)))

    def confusion_matrix(self,y_predict,y_test):
        confusion_matrix(y_test,y_predict)
        return pd.crosstab(y_test, y_predict, rownames=['True'], colnames=['Predicted'], margins=True)

    def runtime_visualization(self,dataset,test_size):

        def _annot_max(x,y, ax=None):
            xmax = x[np.argmax(y)]
            ymax = y.max()
            text= "x={}, y={}\n"\
                  "n={}, d={}".format(xmax, int(ymax),int(xmax*train_size),int(xmax*test_size)+1)
            if not ax:
                ax=plt.gca()
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
            kw = dict(xycoords='data',textcoords="axes fraction",
                    arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
            ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)

        train_size = 1 - test_size

        # Define X and Y variable data
        x = np.arange(len(dataset)+1)
        y = []

        for element in x:
            training_size = element * train_size
            testing_size = element * test_size
            
            element = int(training_size) * (int(testing_size)+1)
            y.append(element)

        y = np.asarray(y)
        fig, ax = plt.subplots()
        ax.plot(x, y)

        plt.xlabel("Training Sample")  # add X-axis label
        plt.ylabel("Calculation of Euclidian Distance Counter")  # add Y-axis label
        plt.title("Runtime Complexity of Bruteforce kNN O(n*d)")  # add title

        _annot_max(x,y)
        fig.tight_layout()

        return plt

# Driver code
def main() :
      
    # Importing dataset
    df = pd.read_csv("diabetes.csv")
    print(df.shape)
    
    X = df.iloc[:,:-1].values
  
    Y = df.iloc[:,-1:].values
      
    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split( 
      X, Y, test_size = 0.2, random_state = 0)

    # scaling data/standardizing data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # calculate value of k(n_neighbors)
    k = math.sqrt(len(Y_test))
    k = int(k)
    if k % 2 == 0:
        k -= 1
    
    # Model training
    model = KNNClassifier(K = k)
      
    model.fit(X_train, Y_train)
      
    # Prediction on test set
    Y_pred,counter = model.predict(X_test)

    print("Dimension in training set: ",X_train.shape[0])
    print("Train test sample: ",X_test.shape[0])
    #O(nd)
    #n = train test sample
    #d = dimension in training set
    print("O(n*d) where n = {test} and d = {train} =".format(test =X_test.shape[0],train=X_train.shape[0]), counter)
 
    # measure performance
    correctly_classified = 0

      
    for i in range( np.size( Y_pred ) ) :
        if Y_test[i] == Y_pred[i] :
            correctly_classified = correctly_classified + 1
          
    print("Accuracy on test set by our model:  ", (correctly_classified / i ) * 100)
    
    print(model.confusion_matrix(Y_pred,Y_test.flatten()))

    runtime_visualization = model.runtime_visualization(df,0.2)
    runtime_visualization.show()

if __name__ == "__main__" : 
      
    main()