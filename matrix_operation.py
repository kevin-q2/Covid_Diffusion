import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression
import random
import copy

################################################################################################################
#
# A class that performs some matrix operations and methods specifically on pandas dataframes.
# I made this to make some data cleaning operations (specifically ones that we find helpful for
# our project) a bit easier and more versatile.
#
# For example) If I pass in my original dataframe I can then simply normalize the values
#               as well as perfrom things like isotonic regression (very helpful in our case)
#
# INPUT: 
#   dataframe - a pandas dataframe which you wish to work with
#
# ATTRIBUTES:
#   dataframe - the original/modified pandas dataframe
#   array - data in numpy array form
#
# METHODS:
#   known, unknown, drop_val_row, drop_val_col, normalizer, population_normalizer,
#   moving_average, new_case_calc, pairwise_distance, non_mon, is_row_inc, is_col_inc
#   known_for_iso, iso, rank_approx
#
#   details and descriptions for each method are found below 
#   
#
#################################################################################################################



class mat_opr:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.array = self.dataframe.values.tolist()


    def known(self, unknowns = None):
        # returns a list of tuples with indices for known entries in the matrix
        # unkowns can take the value 0 in order to represent zeros as unkowns

        indices = []
        for i in range(len(self.array)):
            for j in range(len(self.array[i])):
                if self.array[i][j] != unknowns:
                    if not self.array[i][j] is unknowns:
                        if not np.isnan(self.array[i][j]):
                            indices.append((i,j))

        return indices
    
    
    def unknown(self, unknowns = None):
        # returns a list of tuples with indices for unknown entries in the matrxi
        # where unknowns are represented as the value of unknowns
        
        indices = []
        for i in range(len(self.array)):
            for j in range(len(self.array[i])):
                if self.array[i][j] == unknowns:
                    indices.append((i,j))
                elif unknowns is None:
                    if self.array[i][j] is None:
                        indices.append((i,j))
                elif np.isnan(unknowns):
                    if np.isnan(self.array[i][j]):
                        indices.append((i,j))

        return indices
        

    def drop_val_row(self, val=0):
        # drops any row that contains all zeros or all of the value provided in val
        newframe = self.dataframe.loc[(self.dataframe!=val).any(axis=1)]
        self.dataframe = newframe
        self.array = self.dataframe.values.tolist()


    def drop_val_col(self, val=0):
        # drops any column that contains all zeros or all of the value provided in val
        newframe = self.dataframe.loc[:, (self.dataframe!=val).any(axis=0)]
        self.dataframe = newframe
        self.array = self.dataframe.values.tolist()
        
    

    def normalizer(self, maxer=False):
        # This function normalizes a dataframe by dividing each column by its max value if maxer == True
        # if maxer == False it will divide each column by the last entry in the column
        # returns a new matrix operation object

        norm = self.dataframe.copy(deep=True)
        if maxer:
            for i in norm.columns:
                norm[i] /= norm[i].max()
        else:
            for i in norm.columns:
                norm[i] /= norm[i].iloc[-1]

        return mat_opr(norm)


    
    def population_normalizer(self, pop_frame, level = None):
        #Normalized based on populations. i.e. for schools divide all cases by total enrollment
        # AS input it takes a dictionary where each key is a column name
        # and each value is the population to divide by 
        # ex) {'Alaska':731545}
        
        # Note: this can also be generalized to normalize based on any input values
        # Returns a new matrix operation object

        norm = self.dataframe.copy(deep=True)
        droppers = []
        for i in norm.columns:
            try:
                norm[i] /= pop_frame[i]
            except:
                if level is not None:
                    try:
                        norm[i] /= pop_frame[i[level]]
                    except KeyError:
                        print(i)
                        droppers.append(i)
                else:
                    pass
        norm = norm.drop(droppers, axis = 1)                
        return mat_opr(norm)
    
    

    def moving_average(self, period):
        # calcuates a sliding window average with given period
        # returns a new matrix operation object
        
        move = {}
        for col in self.dataframe.columns:
            move[col] = self.dataframe.loc[:,col].rolling(window=period).mean()

        roller = pd.DataFrame.from_dict(move)
        roller.index = self.dataframe.index

        return mat_opr(roller)

        

    def new_case_calc(self):
        # This method returns a new matrix object where the cases are considered based on daily increase
        # instead of a cumulative count

        new_frame_dict = {}

        for col in self.dataframe.columns:
            bosu = self.dataframe[col]

            new_cases = []
            for i in range(len(bosu)):
                if i == 0:
                    new_cases.append(0)
                else:
                    diff = bosu.iloc[i] - bosu.iloc[i-1]
                    new_cases.append(diff)

            new_frame_dict[col] = new_cases

        news = pd.DataFrame(new_frame_dict)
        news.index = self.dataframe.index
        return mat_opr(news)
    
    

    def pairwise_distance(self, axis=1):
        # Calculates the pairwise distance between every row or column and all possible pairs
        # axis = 1 --- columns
        # axis = 0 --- rows
        # returns a pandas dataframe of the distances 
        # The returned dataframe is square in size 

        if axis == 1:
            dist = {col:[] for col in self.dataframe.columns}
            for col in self.dataframe.columns:
                for pair in self.dataframe.columns:
                    col_arr = np.array(self.dataframe.loc[:,col])
                    pair_arr = np.array(self.dataframe.loc[:,pair])
                    dist[col].append(np.linalg.norm(col_arr - pair_arr))
            
            return pd.DataFrame.from_dict(dist)

        else:
            dist = {row:[] for col in self.dataframe.index}
            for row in self.dataframe.index:
                for pair in self.dataframe.index:
                    row_arr = np.array(self.dataframe.loc[row,:])
                    pair_arr = np.array(self.dataframe.loc[pair,:])
                    dist[row].append(np.linalg.nor(row_arr - pair_arr))

            return pd.DataFrame.from_dict(dist, orient='index')



    def non_mon(self, dicter):
        # is_col_inc() and is_row_inc() return a dictionary of indices that violate -- {column: [rows]}
        # this is a function to compute the fraction of points that violate monotonicity

        total = self.dataframe.size
        frac = 0
        for i in dicter.values():
            frac += len(i)
            
        return frac/total
    

    def is_row_inc(self, printy=True):
        # Tests if the data frame is row increasing
        # prints results if printy is set to True
        # returns a dictionary of indices where the dataframe is not increasing
        # dictionary structure: {row index: [column indices]}
        # DOES not include cases where the entry is 0 or unkown

        non_inc = {}
        for i in range(len(self.array)):
            last = self.array[i][0]
            spots = []
            for j in range(len(self.array[0])):
                if self.array[i][j] != 0 and not np.isnan(self.array[i][j]):
                    if self.array[i][j] < last:
                        spots.append(j)
                    last = self.array[i][j]

            if len(spots) != 0:
                non_inc[i] = spots

        if printy == True:
            print(str(self.non_mon(non_inc) * 100) + " percent of the data points are non-increasing")
            
        return non_inc
    

    def is_col_inc(self):
        # Tests if the data frame is column increasing
        # prints results if printy is set to True
        # returns a dictionary of indices where the dataframe is not increasing
        # dictionary structure: {column index: [row indices]}
        # DOES not include cases where the entry is 0 or unkown

        non_inc = {}
        for i in range(len(self.array[0])):
            last = self.array[0][i]
            spots = []
            for j in range(len(self.array)):
                if self.array[j][i] != 0 and not np.isnan(self.array[j][i]):
                    if self.array[j][i] < last:
                        spots.append(j)
                    last = self.array[j][i]

            if len(spots) != 0:
                non_inc[i] = spots


        return non_inc, self.non_mon(non_inc) * 100


    def known_for_iso(self, axis, unknowns):
        # a helper function for isotonic regression which simply takes the list of known tuple indices
        # and turns it into form -- {column index : [row indices]} if axis == 1
        # or {row index : [column indices]} if axis == 0
        
        knowns = self.known(unknowns)

        if axis == 1:
            known_dict ={ite:[] for ite in range(len(self.dataframe.columns))}
            for i in knowns:
                known_dict[i[1]].append(i[0])

        else:
            known_dict ={ite:[] for ite in range(len(self.dataframe.index))}
            for i in knowns:
                known_dict[i[0]].append(i[1])

        return known_dict
    
    

    def iso(self, axis=1, unknowns = 0):
        # performs isotonic regression ONLY for known data values
        # and ONLY on columns or rows where there are non-increasing points
        # row-wise (axis = 0) or column-wise (axis = 1)
        # unknowns should be 0 or None or similar
        
        # returns a new matrix operation object where isotonic regression has been performed

        tonic = copy.deepcopy(self.array) # returns a new isotonic matrix
        known_dict = self.known_for_iso(axis, unknowns)
        
        if axis == 1:
            increase_dict, non_increase_percent = self.is_col_inc()
        else:
            increase_dict = self.is_row_inc()

        # increase_dict tells me where things arent increasing (from is_row_inc() or is_col_inc())
        if axis == 1:
            for i in range(len(tonic[0])):
                try:
                    # if i is a key in increase dict then this column needs regression
                    # else just pass to the next column
                    tester = increase_dict[i]

                    X = known_dict[i]

                    if X != []:
                        initial_vals = [tonic[j][i] for j in X]

                        # Use the initial values to fit the model and then predict what the decreasing ones should be
                        iso = IsotonicRegression(out_of_bounds='clip').fit(X,initial_vals)
                        predictions = iso.predict(range(len(tonic)))

                        # put everything back:
                        for row in range(len(predictions)):
                            tonic[row][i] = predictions[row]
                except:
                    pass

        else:
            # same thing but with rows
            for i in range(len(tonic)):
                try:
                    tester = increase_dict[i]
                    X = known_dict[i]

                    if X != []:

                        initial_vals = [tonic[i][j] for j in X]

                        # Use the initial values to fit the model and then predict what the decreasing ones should be
                        iso = IsotonicRegression(out_of_bounds='clip').fit(X,initial_vals)
                        predictions = iso.predict(range(len(tonic[i])))

                        # put everything back:
                        tonic[i] = predictions

                except:
                    pass

        newframe = pd.DataFrame(tonic)
        newframe.columns = self.dataframe.columns
        newframe.index = self.dataframe.index


        if unknowns == 0:
            # Isotonic outputs NaN values, replace them with zeros
            newframe = newframe.fillna(0)

        return mat_opr(newframe)
    
    

    def rank_approx(self, percent = 0.8):
        # approximates the rank of a matrix by summing the squares of the singular values
        # once the sum has surpassed the percent threshold (given as input)
        # the rank is assumed to be found and then returned

        u,s,vt = np.linalg.svd(self.array)
        denom = 0
        for i in s:
            denom += i**2

        numer = 0
        ratio = 0
        ranker = 0
        while ratio < percent and ranker < len(s):
            numer += s[ranker]**2
            ratio = numer/denom
            ranker += 1

        return ranker
