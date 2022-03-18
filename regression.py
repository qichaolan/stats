import argparse
import numpy as np
import pandas as pd
import yaml
from os.path import isfile
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from itertools import combinations

class Regression(object):
    def __init__(self):
        pass

    def __read_training_file(self, csv_file, y_col):
        ''' read training data from csv file, will skip the row if y_col is empty or <= 0

        Returns:
            a two-dimensional panda dataframe
        '''
        return pd.concat([chunk[chunk[y_col[0]] is not None and chunk[y_col[0]] > 0] for chunk in pd.read_csv(csv_file, iterator=True, chunksize=1000)])

    def __read_variable_file(self, var_file):
        ''' read training variables from a YAML file

        Returns:
            x_cols: a list of variable names for x cols
            y_cols: a list of variable names for y col, it has one variable ONLY
        '''

        x_cols =""
        y_cols = ""

        # Read YAML file
        with open(var_file, 'r') as stream:
            try:
                vars = yaml.safe_load(stream)
                for type in vars:
                    if type == "x_cols":
                        x_cols = vars[type]
                    
                    if type == "y_cols":
                        y_cols = vars[type]
            except yaml.YAMLError as exc:
                print(exc)

        return x_cols, y_cols

    def train_best_regression_model(self, training_file, var_file, least_var):
        ''' main entry of this class, run mutiple regression training

        Args:

        Returns:
           
        '''

        #print(training_file, out_file, var_file, least_var, top, eval_file)

        # read x and y columns from the YAML file
        x_cols, y_col = self.__read_variable_file(var_file)
        
        #if number of variables is greater than required, then returns True
        assert len(x_cols) >= least_var

        #if number of Y variable == 1, then returns True
        assert len(y_col) == 1

        # read training data
        df = self.__read_training_file(training_file, y_col)
        print (df.shape)
    
        highest_r2adj = float('-inf')
        best_model = []
        #stepwise training
        for r in range(least_var, len(x_cols)+1):
            a = combinations(x_cols, r)
            b = [','.join(i) for i in a]
            for x in b:
                intercept, coefs, rsquared, adj_squared, rMSE, MSE = self.__train(df, x.split(","), y_col)
                
                # no need to save the results if its adj_squared is lower than current best model
                if highest_r2adj > adj_squared: continue

                highest_r2adj = adj_squared
                best_model = [intercept, coefs, rsquared, adj_squared, rMSE, MSE] 

        return best_model

    def __train(self, df, x_cols, y_col):
        """
            
        """
        # define dataset

        X = df[x_cols] 
        y = df[y_col[0]]  

        # fit the model
        regr = LinearRegression()
        regr.fit(X, y)        # get importance

        # Regression Metrics
        yhat = regr.predict(X)
        SS_Residual = sum((y-yhat)**2)       
        SS_Total = sum((y-np.mean(y))**2)     
        
        r_squared = 1 - (float(SS_Residual))/SS_Total
        adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
     
        rMSE = mean_squared_error(y, yhat, squared=False)
        MSE = mean_squared_error(y, yhat, squared=True)

        return [regr.intercept_,
            regr.coef_,
            round(r_squared,4),
            round(adjusted_r_squared,4),
            round(rMSE,4),
            round(MSE,4)]

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument("-t", "--train", type=str, required=True,
        help="Path to a csv file to store the training data set")

    parser.add_argument("-e", "--eval", type=str, 
        help="Path to a csv file to store the evaluation/test data set")

    parser.add_argument("-o", "--out", type=str,required=True,
        help="Path to a path and filename to store output")

    parser.add_argument("-l", "--least", type=int, default=1,
        help="Specify the least variables you want to use, default 1")

    parser.add_argument("-b", "--best", type=int, default=1,
        help="Specify a number which indicate the top x models you want to save, default 10")

    parser.add_argument("-v", "--variables", type=str, required=True,
        help="Specify a YMAL file to store training variables")

    args = parser.parse_args()

    if not isfile(args.train):
        print("Training file not exists! Aborting.")
        exit(-1)

    if not isfile(args.variables):
        print("Variables file not exists! Aborting.")
        exit(-1)
 
    if isfile(args.out):
        print("Output file exists already, pls change one! Aborting.")
        exit(-1)

    d = Regression()
    
    rst = d.train_best_regression_model(args.train, args.variables, args.least)

    print (rst)

    return rst

if __name__ == '__main__':
    main()