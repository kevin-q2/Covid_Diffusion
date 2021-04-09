import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import  NMF
import random
import copy
#from NonnegMFPy import nmf

# Note lmafit.py needs to be in the same directory for now
from lmafit import lmafit_mc_adp

# A class that performs some matrix operations and methods specifically on
# Pandas dataframes

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
                if unknowns == 0:
                    if self.array[i][j] != 0:
                        indices.append((i,j))
                else:
                    if not(self.array[i][j] is None or self.array[i][j] is np.nan):
                        indices.append((i,j))

        return indices
    
    def unknown(self, unknowns = None):
        if unknowns == None:
            self.dataframe = self.dataframe.fillna(0)
            self.array = self.dataframe.values.tolist()
        elif unknowns == 0:
            self.dataframe = self.dataframe.replace(0, np.nan)
            self.array = self.dataframe.values.tolist()


    def drop_zero_rows(self, val=0):
        # drops any row that contains all zeros or all of the value provided in val
        newframe = self.dataframe.loc[(self.dataframe!=val).any(axis=1)]
        self.dataframe = newframe
        self.array = self.dataframe.values.tolist()

    def drop_zero_cols(self, val=0):
        # drops any column that contains all zeros or all of the value provided in val
        newframe = self.dataframe.loc[:, (self.dataframe!=val).any(axis=0)]
        self.dataframe = newframe
        self.array = self.dataframe.values.tolist()

    def hide_rows(self, percent, non_random=None):
        # returns a new mat_opr object with a specified percent of the original rows randomly hidden
        # OR if non_random (a list of indexes to drop from) is defined drop randomly from that list

        new_arr = self.dataframe.copy(deep=True)

        num_hide = int(len(new_arr.index)*percent)

        if not non_random is None:
            if num_hide > len(non_random):
                raise IndexError("Trying to hide too many values")
            else:
                to_hide = random.sample(non_random, num_hide)

                for t in to_hide:
                    new_arr = new_arr.drop(t, axis=0)
                
        else:
            to_hide = random.sample(list(new_arr.index), num_hide)

            for t in to_hide:
                new_arr = new_arr.drop(t, axis=0)

        return mat_opr(new_arr)
    
    def interpolater(self, method='linear'):
        # interpolates between datapoints with a specified method
        new_inter = self.dataframe.copy(deep=True)
        new_inter = new_inter.replace(0, np.nan)
        for col in new_inter.columns:
            new_inter[col] = new_inter[col].interpolate(method=method)

        new_inter = new_inter.fillna(0)
        return mat_opr(new_inter)

    def hide_cols(self, percent):
        # returns a new mat_opr object with a specified percent of the original columns randomly hidden
        new_arr = self.dataframe.copy(deep=True)

        num_hide = int(len(new_arr.columns)*percent)
        to_hide = random.sample(list(new_arr.columns), num_hide)

        for t in to_hide:
            new_arr = new_arr.drop(t, axis=1)

        return mat_opr(new_arr)

    def hide_entries(self, percent, val = None):
        # Hides a percent of the known entries in the dataframe
        # returns a new hidden mat_opr object and a dictionary for indexes and values
        # of the entries that were hidden

        # val takes a value that "represents" a hidden value
        # ex) None or 0

        new_arr = self.dataframe.copy(deep=True)
        knowns = self.known(val)

        num_hide = int(len(knowns)*percent)
        to_hide = random.sample(knowns, num_hide)

        hiders = {}
        for t in to_hide:
            hiders[t] = new_arr.iloc[t[0],t[1]]
            new_arr.iloc[t[0],t[1]] = val

        return mat_opr(new_arr), hiders

    def error(self, hidden_vals, newframe):
        # computes error between hidden vals and a supplied data object

        # hidden_vals is a dictionary with structure {(x,y): value}
        # new frame is the dataframe after doing matrix completion etc.
        
        # Keeping things simple by doing mean absolute error for now
        summ = 0
        for h in hidden_vals.keys():
            frame_val = newframe.iloc[h[0],h[1]]
            if frame_val is None or np.isnan(frame_val):
                frame_val = 0
            err = abs(frame_val - hidden_vals[h])
            summ += err
            
        mean_absolute = summ/len(hidden_vals)
        
        #print("The computed matrix has an average error of " + str(mean_absolute))
        #print()
        return mean_absolute

    def mean_square_error(self, newframe, unknowns):
        # computes mean squared error between the original matrix
        # and a new one: newframe
        # 
        # Note that this only computes the error on known values 
        known_ind = self.known(unknowns)
        meaner = 0
        for ind in known_ind:
            orig = self.dataframe.iloc[ind[0],ind[1]]
            newn = newframe.dataframe.iloc[ind[0], ind[1]]
            meaner += (orig - newn) ** 2

        return meaner/len(known_ind)

    def hidden_tester(self, trials = [], method = 'nmf', ranker = 0, isotonic = False):
        # hides a random set of entries then performs a matrix completion method ('nmf' or 'lmafit')
        # along with isotonic regression if set to true

        # The function will then compute the error between the completed matrix and the hidden values
        # This can be over a set of different trials:
        # trials should be a list of percents [0.33, 0.25, ...] corresponding to how much data is hidden
        trials.sort()
        complete_results = []
        iso_results = []
        if ranker == 0:
            ranker = self.rank_approx()

        for i in trials:
            # Make the new hidden matrix and record the values that were hidden
            hidden_matrix, hidden_values = self.hide_entries(i, 0)
            
            # Do lmafit
            if method == 'lmafit':
                X,Y,other = hidden_matrix.lmafitter(rank = ranker, val = 0)
                complete = pd.DataFrame(np.dot(X,Y))
                complete = mat_opr(complete)
            elif method == 'nmf':
                complete = hidden_matrix.missing_nmf(ranker)
            else:
                raise error("invalid method name")
            
            
            # Optional: do isotonic regression
            if isotonic == True:
                iso_reg = complete.known_iso()
                complete_results.append(self.error(hidden_values, complete.dataframe))
                iso_results.append(self.error(hidden_values, iso_reg.dataframe))
            else:
                complete_results.append(self.error(hidden_values, complete.dataframe))
                
        return complete_results, iso_results

    def normalizer(self, maxer=False):
        # This function normalizes a dataframe by dividing each column by its max value
        # Note: For this project the max value is assumed to be the last (cumulative cases)
        #       So this should work given that isotonic corrects all mistakes beforehand.

        # IF its not the case that the last should be the biggest, setting maxer = True will work

        norm = self.dataframe.copy(deep=True)
        if maxer:
            for i in norm.columns:
                norm[i] /= norm[i].max()
        else:
            for i in norm.columns:
                norm[i] /= norm[i].iloc[-1]

        return mat_opr(norm)


    
    def population_normalizer(self, pop_frame):
        #Normalized based on populations. i.e. for schools divide all cases by total enrollment
        # AS input it takes a dictionary where each key is a column name
        # and each value is the population to divide by 
        # ex) {'Alaska':731545}

        norm = self.dataframe.copy(deep=True)
        for i in norm.columns:
            norm[i] /= pop_frame[i]

        return mat_opr(norm)
    

    def moving_average(self, period):
        move = {}
        for col in self.dataframe.columns:
            move[col] = self.dataframe.loc[:,col].rolling(window=3).mean()

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
        knowns = self.known(unknowns)

        if axis == 1:
            known_dict ={ite:[] for ite in range(len(self.dataframe.columns))}
            for i in knowns:
                #try:
                known_dict[i[1]].append(i[0])
                #except:
                    #known_dict[i[1]] = [i[0]]
        else:
            known_dict ={ite:[] for ite in range(len(self.dataframe.index))}
            for i in knowns:
                #try:
                known_dict[i[0]].append(i[1])
                #except:
                    #known_dict[i[0]] = [i[1]]
        return known_dict

    def iso(self, axis=1, unk = 'No'):
        # performs isotonic regression row-wise (axis = 0) or column-wise (axis = 1)
        tonic = copy.deepcopy(self.array) # returns a new isotonic matrix

        # either use a value for unknowns or just do isotonic with all present values
        if unk == 0 or unk is None:
            known_dict = self.known_for_iso(axis, unk)
        else:
            known_dict = None

        # dat dict tells me where things arent increasing (from is_row_inc() or is_col_inc())
        if axis == 1:
            if known_dict is None:
                for i in range(len(tonic[0])):
                    initial_vals = [tonic[j][i] for j in range(len(tonic))]
                    X = list(range(len(initial_vals)))

                    # Use the initial values to fit the model and then predict what the decreasing ones should be
                    iso = IsotonicRegression(out_of_bounds='clip').fit(X,initial_vals)
                    predictions = iso.predict(range(len(tonic)))

                    # put everything back:
                    for row in range(len(predictions)):
                        tonic[row][i] = predictions[row]

            else:
                for i in range(len(tonic[0])):
                    X = known_dict[i]
                    initial_vals = [tonic[j][i] for j in X]

                    # Use the initial values to fit the model and then predict what the decreasing ones should be
                    iso = IsotonicRegression(out_of_bounds='clip').fit(X,initial_vals)
                    predictions = iso.predict(range(len(tonic)))

                    # put everything back:
                    for row in range(len(predictions)):
                        tonic[row][i] = predictions[row]


        else:
            if known_dict is None:
                for i in range(len(tonic)):
                    initial_vals = [tonic[i][j] for j in range(len(tonic[0]))]
                    X = list(range(len(initial_vals)))

                    # Use the initial values to fit the model and then predict what the decreasing ones should be
                    iso = IsotonicRegression(out_of_bounds='clip').fit(X,initial_vals)
                    predictions = iso.predict(range(len(tonic)))

                    # put everything back:
                    tonic[i] = predictions

            else:
                for i in range(len(tonic)):
                    X = known_dict[i]
                    initial_vals = [tonic[i][j] for j in X]

                    # Use the initial values to fit the model and then predict what the decreasing ones should be
                    iso = IsotonicRegression(out_of_bounds='clip').fit(X,initial_vals)
                    predictions = iso.predict(range(len(tonic)))

                    # put everything back:
                    tonic[i] = predictions

        newframe = pd.DataFrame(tonic)
        newframe.columns = self.dataframe.columns
        newframe.index = self.dataframe.index
        return mat_opr(newframe)

    def known_iso(self, axis=1, unknowns = 0):
        # performs isotonic regression ONLY for known data values
        # and ONLY on columns where there are non-increasing points
        # row-wise (axis = 0) or column-wise (axis = 1)
        # unknowns should be 0 or none

        tonic = copy.deepcopy(self.array) # returns a new isotonic matrix
        known_dict = self.known_for_iso(axis, unknowns)
        if axis == 1:
            increase_dict, non_increase_percent = self.is_col_inc()
        else:
            increase_dict = self.is_row_inc()

        # dat dict tells me where things arent increasing (from is_row_inc() or is_col_inc())
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

    def lmafitter(self, rank = None, val=0):
        # Perfroms low-rank matrix completion using the methods from lmafit.py
        # val takes the value of unknowns ex) either 0 or None
        # rank takes an approximate rank for the matrix
        if rank is None:
            rank = self.rank_approx()

        #First need to make the arrays needed
        known_seq = [[],[]]
        known_values = []
        known_ind = self.known(val)
        for i in known_ind:
            known_seq[0].append(i[0])
            known_seq[1].append(i[1])
            known_values.append(self.array[i[0]][i[1]])

        # something here has been deprecated. Not sure how to fix yet.
        known_indices = [tuple(known_seq[0]), tuple(known_seq[1])]
        known_values = [tuple(known_values)]




        X,Y,out = lmafit_mc_adp(len(self.array),len(self.array[0]),rank,known_indices,known_values)

        return X,Y,out

    def sci_nmf(self, components = 2, procedure=None, separate = False, max_iter=1000):
        # Performs non-negative matrix factorization
        # procedure takes one of the initial methods of approximation listed in the parameters for nmf here:
        # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

        arr = np.array(self.array)
        model = NMF(n_components= components, init=procedure, random_state=0, max_iter=1000, tol=1e-10)
        W = model.fit_transform(arr)
        H = model.components_

        # Return either multiplied together as a new object OR as decomposed matrices
        if separate == False:
            tor = mat_opr(pd.DataFrame(np.dot(W,H)))
            tor.dataframe.columns = self.dataframe.columns
            tor.dataframe.index = self.dataframe.index
            return tor
        else:
            return W, H


# taking this out for now because I'm not using it.
"""
    def missing_nmf(self, components, max_iterations = 1000, separate = False):
        # this is just a different implementation of nmf that is able to handle
        # missing or unknown data
        # https://www.guangtunbenzhu.com/nonnegative-matrix-factorization

        mask = np.array(self.dataframe.astype(bool).values.tolist())
        arr = np.array(self.array)

        #This is just so I have a consistent random state
        mod = np.random.RandomState(101) #weird that something like 39 throws everything off
        # 301 is good for 2
        dub = mod.rand(arr.shape[0], components)
        vee = mod.rand(components, arr.shape[1])

        model = nmf.NMF(X=arr, W=dub, H=vee, M=mask, n_components=components)
        model.SolveNMF(sparsemode=True, maxiters=max_iterations)
        w = model.W
        h = model.H

        if separate == False:
            tor = mat_opr(pd.DataFrame(np.dot(w,h)))
            tor.dataframe.columns = self.dataframe.columns
            tor.dataframe.index = self.dataframe.index
            return tor
        else:
            return w,h
"""
            


"""
# a function for plotting basis vectors that I haven't implemented yet...
def plot_bases(data_obj, ranky):
    w,h = data_obj.missing_nmf(ranky,separate = True)
    bases = pd.DataFrame(w)
    
    for i in bases.columns:
        bases[i].plot()
        plt.title("basis " + str(i))
        plt.show()

"""