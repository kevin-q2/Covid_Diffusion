import os
import sys
import glob
import re
import pandas as pd
import datetime as dt
import random
import numpy as np
import copy


cwd = os.getcwd()
par = os.path.dirname(cwd)
parpar = os.path.dirname(par)
sys.path.append(par)
from matrix_operation import mat_opr


###################################################################################################################
#
#
# A set of classes with methods used to clean and save John's Hopkins STATE LEVEL Case data
# Note that this is specifically designed on my file system, I haven't yet extended it to a more general case
# In my case, I have the John's hopkins git hub repository cloned (https://github.com/CSSEGISandData/COVID-19)
# And I need to pull from it before updating everything here
#
# state_data is a class to import state case data
#
# state_df contains the case data and get_state_data() collects and cleans it
# Setting get_state_dat to True allows this collecting/cleaning process to happen
# otherwise it will just read a preloaded version
# 
#
# MORE WORK TO BE DONE ONCE I GENERALIZE THIS
#
###################################################################################################################

class state_data(mat_opr):
    def __init__(self, get_state_dat = False, saver = False):


         # State Case data
        if get_state_dat:
            self.state_df = self.get_state_data()
            if saver:
                self.state_df.to_csv(os.path.join(par, "collected_data/state_dataset.csv"))
        else:
            self.state_df = pd.read_csv(os.path.join(par, "collected_data/state_dataset.csv"), index_col = 0)

        # initialize within the matrix operation framework
        self.state_cases = super().__init__(self.state_df)



    def get_state_data(self):

        # path to John Hopkins dataset
        state_path = os.path.join(parpar, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports_us', '')

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

        # Some locations not being considered for now
        state_cases.pop('Recovered')
        state_cases.pop('American Samoa')
        state_cases.pop('Diamond Princess')
        state_cases.pop('Grand Princess')
        state_cases.pop('Northern Mariana Islands')
        state_cases.pop('Virgin Islands')
        state_cases.pop('Guam')

        cumul = pd.DataFrame(data=state_cases, index=ind)
        
        days = []
        for ind in cumul.index:
            days.append(dt.datetime.strptime(ind, "%m-%d-%Y"))
        
        cumul.index = days
        cumul = cumul.sort_index()

        return cumul





    def get_other_data(self):

        # path to John Hopkins dataset
        state_path = os.path.join(parpar, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports', '')

        fnames = sorted(glob.glob(state_path+'*.csv'))
        ind = []
        state_cases = {}
        for f in range(len(fnames)):
            date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)
            date_frame = pd.read_csv(fnames[f], index_col=0)

            try:
                date_frame = date_frame.loc[date_frame.Country_Region == 'US']

                for d in date_frame.Province_State.value_counts().index:
                    statey = date_frame.loc[date_frame.Province_State == d]
                    try:
                        state_cases[d].append(statey['Confirmed'].sum())
                    except:
                        state_cases[d] = [statey['Confirmed'].sum()]
                ind.append(date)
            except:
                pass


        # Some locations not being considered for now
        nos = ['Recovered', 'American Samoa', 'Diamond Princess', 'Grand Princess', 'Northern Mariana Islands',
         'Virgin Islands', 'Guam', 'Wuhan Evacuee']
        for lock in nos:
            try:
                state_cases.pop(lock)
            except:
                pass

        cumul = pd.DataFrame(data=state_cases, index=ind)
        
        days = []
        for ind in cumul.index:
            days.append(dt.datetime.strptime(ind, "%m-%d-%Y"))
        
        cumul.index = days
        cumul = cumul.sort_index()

        return cumul



        

    def get_incidence_rate(self):

        # path to John Hopkins dataset
        state_path = os.path.join(parpar, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports_us', '')

        fnames = sorted(glob.glob(state_path+'*.csv'))
        ind = []
        state_rates = {}
        for f in range(len(fnames)):
            date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)
            ind.append(date)
            try:
                date_frame = pd.read_csv(fnames[f], index_col="Province_State")

                for d in date_frame.index:
                    if f == 0:
                        state_rates[d] = []
                    try:
                        state_rates[d].append(date_frame.loc[d,"Incident_Rate"])
                    except KeyError:
                        pass
            except:
                print(fnames[f])


        # Some locations not being considered for now
        nos = ['Recovered', 'American Samoa', 'Diamond Princess', 'Grand Princess', 'Northern Mariana Islands',
         'Virgin Islands', 'Guam', 'Wuhan Evacuee']
        for lock in nos:
            try:
                state_rates.pop(lock)
            except:
                pass

        cumul = pd.DataFrame(data=state_rates, index=ind)
        
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
                self.test_df.to_csv(os.path.join(par, "collected_data/state_testing.csv"))
        else:
            self.test_df = pd.read_csv(os.path.join(par, "collected_data/state_testing.csv"), index_col = 0)

        # initialize within the matrix operation framework
        self.state_testing = super().__init__(self.test_df)

    def get_test_data(self):
        # same process but with testing rates

        #path to johns hopkins data
        state_path = os.path.join(parpar, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports_us', '')

        fnames = sorted(glob.glob(state_path+'*.csv'))
        ind = []
        test_rates = {}
        for f in range(len(fnames)):
            date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)
            ind.append(date)
            date_frame = pd.read_csv(fnames[f], index_col="Province_State")
            if 'People_Tested' in date_frame.columns:
                caller = 'People_Tested'
            else:
                caller = 'Total_Test_Results'
            for d in date_frame.index:
                try:
                    test_rates[d].append(date_frame.loc[d,caller])
                except:
                    test_rates[d] = [date_frame.loc[d,caller]]

        #Some locations not being considered for now

        test_rates.pop('Recovered')
        test_rates.pop('American Samoa')
        test_rates.pop('Diamond Princess')
        test_rates.pop('Grand Princess')
        test_rates.pop('Northern Mariana Islands')
        test_rates.pop('Guam')
        test_rates.pop('Virgin Islands')
        cumul = pd.DataFrame(data=test_rates, index=ind)
        
        days = []
        for ind in cumul.index:
            days.append(dt.datetime.strptime(ind, "%m-%d-%Y"))
        
        cumul.index = days
        cumul = cumul.sort_index()

        return cumul

    def get_positivity_rates(self):
        # same process but with testing rates

        #path to johns hopkins data
        state_path = os.path.join(parpar, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports_us', '')

        fnames = sorted(glob.glob(state_path+'*.csv'))
        ind = []
        inc_rates = {}
        test_rates = {}
        for f in range(len(fnames)):
            date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)
            ind.append(date)
            date_frame = pd.read_csv(fnames[f], index_col="Province_State")

            for d in date_frame.index:
                if f == 0:
                    test_rates[d] = []
                try:
                    test_rates[d].append(date_frame.loc[d,"Incident_Rate"]/date_frame.loc[d,"Testing_Rate"])
                except KeyError:
                    pass

    
        # Some locations not being considered for now
        nos = ['Recovered', 'American Samoa', 'Diamond Princess', 'Grand Princess', 'Northern Mariana Islands',
         'Virgin Islands', 'Guam', 'Wuhan Evacuee']
        for lock in nos:
            try:
                test_rates.pop(lock)
            except:
                pass
        cumul = pd.DataFrame(data=test_rates, index=ind)

        days = []
        for ind in cumul.index:
            days.append(dt.datetime.strptime(ind, "%m-%d-%Y"))
        
        cumul.index = days
        cumul = cumul.sort_index()

        return cumul




if __name__ == '__main__':
    # update the datasets.
    state_data(True, True)
    #state_test_data(True, True)