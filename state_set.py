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
    def __init__(self, get_state_dat = False, saver = False):

        # State Case data
        if get_state_dat:
            self.state_df = self.get_state_data()
            if saver:
                self.state_df.to_csv("collected_data/state_dataset.csv")

        else:
            try:
                # If this is being imported as a module
                cwd = os.path.dirname(os.path.realpath(__file__))
                combo = os.path.join(cwd, "collected_data/state_dataset.csv")
                self.state_df = pd.read_csv(combo, index_col = 0)
                print(combo)
            except:
                # If being used in the original file location
                self.state_df = pd.read_csv("collected_data/state_dataset.csv", index_col = 0)

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
        state_path = os.path.join(par, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports', '')

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
                self.test_df.to_csv("collected_data/state_testing.csv")

        else:
            try:
                # If this is being imported as a module
                cwd = os.path.dirname(os.path.realpath(__file__))
                combo = os.path.join(cwd, "collected_data/state_testing.csv")
                self.test_df = pd.read_csv(combo, index_col = 0)
            except:
                # If being used in the original file location
                self.test_df = pd.read_csv("collected_data/state_testing.csv", index_col = 0)


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






class county_data(mat_opr):
    def __init__(self, get_county_dat = False, saver = False):

        # State Case data
        if get_county_dat:
            self.count_df = self.get_county_data()
            if saver:
                self.count_df.to_csv("collected_data/county_dataset.csv")

        else:
            try:
                # If this is being imported as a module
                cwd = os.path.dirname(os.path.realpath(__file__))
                combo = os.path.join(cwd, "collected_data/county_dataset.csv")
                self.count_df = pd.read_csv(combo, index_col = [1,0])
    
            except:
                # If being used in the original file location
                self.count_df = pd.read_csv("collected_data/county_dataset.csv", index_col = [1,0])

        # initialize within the matrix operation framework
        self.county_cases = super().__init__(self.count_df)

    def get_county_data(self):
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
        state_path = os.path.join(par, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports', '')

        fnames = sorted(glob.glob(state_path+'*.csv'))
        ind = []

        # a recent set to help me get consistent county names
        recent = pd.read_csv(state_path + "04-12-2021.csv")
        us = recent.loc[recent.Country_Region == "US"]
        nons = ["Recovered", "Diamond Princess", "Grand Princess", "Northern Mariana Islands", "Guam", "Virgin Islands"]

        multi = []

        for state in us.Province_State.value_counts().index:
            if state in nons:
                pass
            else:
                for county in us.loc[us.Province_State == state].Admin2:
                    #if county != "Unassigned":
                    multi.append((state,county))

        locations = pd.MultiIndex.from_tuples(multi, names=["state", "county"])
        count_cases = [[] for i in range(len(locations))]

        for f in range(len(fnames)):
            date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)

            if dt.datetime.strptime(date, "%m-%d-%Y") >= dt.datetime(2020, 3, 22):
                ind.append(date)
                date_frame = pd.read_csv(fnames[f])
                date_frame = date_frame.loc[date_frame.Country_Region == "US"]
                
                for d in range(len(locations)):
                    try:
                        stater = date_frame.loc[date_frame.Province_State == locations[d][0]]
                        counter = stater.loc[stater.Admin2 == locations[d][1]]
                        count_cases[d].append(counter.loc[counter.index[0],"Confirmed"])
                    except:
                        count_cases[d].append(np.nan)

            else:
                pass


        cumul = pd.DataFrame(data=count_cases, index=locations, columns = ind)

        days = []
        for inder in cumul.columns:
            days.append(dt.datetime.strptime(inder, "%m-%d-%Y"))
        
        cumul.columns = days
        cumul = cumul.reindex(sorted(days), axis = 1)

        return cumul




if __name__ == 'main':
    # update the datasets.
    state_data(True, True)
    state_test_data(True, True)
    county_data(True, True)