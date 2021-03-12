import os
import glob
import re
import pandas as pd
import datetime as dt
import random
import numpy as np
import copy
from matrix_operation import mat_opr

# A few classes to import a cleaned/seperated version of the Johns Hopkins Case + Testing data
#

# state_data is a class to import state case data

# state_df contains the case data and get_state_data() collects and cleans it
# Setting get_state_dat to True allows this collecting/cleaning process to happen
# otherwise it will just read a preloaded version

class state_data(mat_opr):
    def __init__(self, get_state_dat = False ,get_test_dat=False, saver = False):

        # State Case data
        if get_state_dat:
            self.state_df = self.get_state_data()
            if saver:
                self.state_df.to_csv("state_dataset.csv")

        else:
            try:
                # If this is being imported as a module
                cwd = os.path.dirname(os.path.realpath(__file__))
                combo = os.path.join(cwd, "state_dataset.csv")
                self.state_df = pd.read_csv(combo, index_col = 0)
                print(combo)
            except:
                # If being used in the original file location
                self.state_df = pd.read_csv("state_dataset.csv", index_col = 0)

        # initialize within the matrix operation framework
        self.state_cases = super().__init__(self.state_df)

    def get_state_data(self):
        try:
            # if this is being imported as a module
            cwd = os.path.dirname(os.path.realpath(__file__))
            par = os.path.dirname(cwd)
            par = os.path.abspath(par)

        except NameError:
            # else if its being used in its original file location
            cwd = os.getcwd()
            par = os.path.join(cwd, os.pardir)
            par = os.path.abspath(par)

        # path to John Hopkins dataset
        state_path = os.path.join(par, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports_us', '')

        fnames = sorted(glob.glob(state_path+'*.csv'))
        ind = []
        state_cases = {}
        for f in range(len(fnames)):
            date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)
            ind.append(date)
            date_frame = pd.read_csv(fnames[f], index_col="Province_State")

            for d in date_frame.index:
                try:
                    state_cases[d].append(date_frame.loc[d,'Confirmed'])
                except:
                    state_cases[d] = [date_frame.loc[d,'Confirmed']]

        # Unecessary locations
        state_cases.pop('Recovered')
        state_cases.pop('American Samoa')
        state_cases.pop('Diamond Princess')
        state_cases.pop('Grand Princess')
        state_cases.pop('Northern Mariana Islands')
        cumul = pd.DataFrame(data=state_cases, index=ind)
        
        days = []
        for ind in cumul.index:
            days.append(dt.datetime.strptime(ind, "%m-%d-%Y"))
        
        cumul.index = days
        cumul = cumul.sort_index()

        return cumul


# state_test_data is a class to import state testing rate data

# test_df contains the testing rate data and get_test_data() collects and cleans it
# setting get_test_dat to True allows this collecting/ cleaning process to happen
# otherwise it will just read a preloaded version

# Note that testing data is a testing rate -- #of tests per 100,000 persons

# setting saver to True allows saving of the datasets for updates
class state_test_data(mat_opr):
    def __init__(self,get_test_dat=False, saver = False):
        # State testing Rate Data
        if get_test_dat:
            self.test_df = self.get_test_data()
            if saver:
                self.test_df.to_csv("state_testing.csv")

        else:
            try:
                # If this is being imported as a module
                cwd = os.path.dirname(os.path.realpath(__file__))
                combo = os.path.join(cwd, "state_testing.csv")
                self.test_df = pd.read_csv(combo, index_col = 0)
            except:
                # If being used in the original file location
                self.test_df = pd.read_csv("state_testing.csv", index_col = 0)


        # initialize within the matrix operation framework
        self.state_testing = super().__init__(self.test_df)

    def get_test_data(self):
        # same process but with testing rates
        try:
            # if this is being imported as a module
            cwd = os.path.dirname(os.path.realpath(__file__))
            par = os.path.dirname(cwd)
            par = os.path.abspath(par)

        except NameError:
            # else if its being used in its original file location
            cwd = os.getcwd()
            par = os.path.join(cwd, os.pardir)
            par = os.path.abspath(par)

        #path to johns hopkins data
        state_path = os.path.join(par, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports_us', '')

        fnames = sorted(glob.glob(state_path+'*.csv'))
        ind = []
        test_rates = {}
        for f in range(len(fnames)):
            date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)
            ind.append(date)
            date_frame = pd.read_csv(fnames[f], index_col="Province_State")

            for d in date_frame.index:
                try:
                    test_rates[d].append(date_frame.loc[d,'Testing_Rate'])
                except:
                    test_rates[d] = [date_frame.loc[d,'Testing_Rate']]

        #unecessary locations
        test_rates.pop('Recovered')
        test_rates.pop('American Samoa')
        test_rates.pop('Diamond Princess')
        test_rates.pop('Grand Princess')
        test_rates.pop('Northern Mariana Islands')
        cumul = pd.DataFrame(data=test_rates, index=ind)
        
        days = []
        for ind in cumul.index:
            days.append(dt.datetime.strptime(ind, "%m-%d-%Y"))
        
        cumul.index = days
        cumul = cumul.sort_index()

        return cumul