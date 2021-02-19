import os
import glob
import re
import pandas as pd
import datetime as dt
import random
import numpy as np
import copy
from matrix_operation import mat_opr


# A class to simplify importing and combining the NYT and Big10 Case data
# Uses matrix operation class as a parent function
class dataset(mat_opr):
    def __init__(self, get_dat = False, saver = False):
        if get_dat:
            self.big10_df = None
            self.nyt_df = None
            self.get_data()

            # The difference between NYT and Big10 school names
            self.name_translator = {'University of Illinois Urbana-Champaign':'Illinois',
                        'Indiana University Bloomington':'Indiana',
                        'University of Iowa':'Iowa',
                        'University of Maryland, College Park':'Maryland',
                        'Michigan State University':'Michigan State',
                        'University of Minnesota Twin Cities':'Minnesota',
                        'Northwestern University':'Northwestern',
                        'Ohio State University':'Ohio State',
                        'Penn State University':'Penn State',
                        'University of Wisconsin-Madison':'UW-Madison',
                        'University of Michigan':'Michigan',
                        'University of Nebraska-Lincoln':'Nebraska',
                        'Purdue University':'Purdue',
                        'Rutgers University':'Rutgers'}

            self.combined = None
            self.combine_data()

            if saver:
                self.combined.to_csv("combined_dataset.csv")
        
        else:
            try:
                # If this is being imported as a module
                cwd = os.path.dirname(os.path.realpath(__file__))
                combo = os.path.join(cwd, "combined_dataset.csv")
                self.combined = pd.read_csv(combo, index_col = 0)
            except:
                # If being used in the original file location
                self.combined = pd.read_csv("combined_dataset.csv", index_col = 0)

        super().__init__(self.combined)


    def get_data(self):
        # import all the data

        try:
            # if this is being imported as a module
            cwd = os.path.dirname(os.path.realpath(__file__))
            par = os.path.dirname(cwd)
            par = os.path.abspath(par)
            nyt_datapath = os.path.join(cwd, 'UniversityCases', '')
            big10_datapath = os.path.join(par, 'college-covid19-dataset', 'data', '')

        except NameError:
            # else if its being used in its original file location
            cwd = os.getcwd()
            par = os.path.join(cwd, os.pardir)
            par = os.path.abspath(par)

            nyt_datapath = os.path.join(cwd, 'UniversityCases', '')
            big10_datapath = os.path.join(par, 'college-covid19-dataset', 'data', '')

        # For NYT:
        fnames = sorted(glob.glob(nyt_datapath+'*.csv'))
        frames = []
        for f in fnames:
            df = pd.read_csv(f)
            df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%Y')

            df.drop(['Unnamed: 0'], axis = 1, inplace=True)
            df['Cases'] = df['Cases'].apply(lambda x: x.replace(',', '')).astype('int')
            frames.append(df)

        nyt = pd.concat(frames)
        nyt = nyt.sort_values('Date') # Sort for dates because I named them stupidly
        nyt = nyt.drop_duplicates(subset = ['School','Cases']) # drop duplicate rows (problem with nyt)

        self.nyt_df = nyt

        # For BIG10:
        big10 = pd.read_csv(os.path.join(big10_datapath, 'daily.csv'))
        old_cols = big10.columns.values.copy()
        old_cols[0] = 'School'
        big10.columns = old_cols
        big10['Date'] = pd.to_datetime(big10['Date'],format='%Y-%m-%d')
        big10 = big10.sort_values('Date')

        self.big10_df = big10

    def combine_data(self):
        # list of dates for the index
        indexer = []
        start_day = dt.datetime(2020, 9, 8)
        c = 0
        while start_day < dt.datetime.today():
            indexer.append(start_day)
            start_day = start_day + dt.timedelta(days = 1)
            c += 1

        date_index = {}
        for i in range(len(indexer)):
            date_index[indexer[i]] = i

        nyt_schools = list(pd.unique(self.nyt_df.School))
        for i in self.name_translator.keys():
            nyt_schools.remove(i)

        data_dict = {}
        # Add all the big 10 schools
        for i in self.name_translator.values():
            cases = [0 for i in range(len(indexer))]
            school = self.big10_df.loc[self.big10_df.School == i]
            for j in school.Date:
                if j >= dt.datetime(2020,9,8):
                    to_index = date_index[j]
                    cases[to_index] = school.loc[school.Date == j].Confirmed.iloc[0]

            data_dict[i] = cases


        # Add all the NYT
        for i in nyt_schools:
            cases = [0 for i in range(len(indexer))]
            school = self.nyt_df.loc[self.nyt_df.School == i]
            for j in school.Date:
                to_index = date_index[j]
                cases[to_index] = school.loc[school.Date == j].Cases.iloc[0]

            data_dict[i] = cases

        # Throw everything into a data frame and drop all the zero rows:
        incomplete_matr = pd.DataFrame.from_dict(data_dict)
        incomplete_matr.index = indexer

        no_zero = incomplete_matr.loc[(incomplete_matr!=0).any(axis=1)]

        self.combined = no_zero
    
    def drop_bad_rows(self):
        # Professor from big10 collection stopped collecting so I need to drop
        # some rows where he started dropping off
        listy = list(self.dataframe.index)
        cop = self.dataframe.copy(deep=True)

        for i in self.dataframe.index:
            if (self.dataframe.loc[i]!=0).sum() <= 10:
                cop = cop.drop(i, axis=0)

        return mat_opr(cop)


"""
def new_case_calc(data_obj, name=None, locr=None):
    if locr is None and name is None:
        return "need more arguments"
    
    elif locr is None:
        bosu = data_obj.dataframe[name]
    else:
        bosu = data_obj.dataframe.iloc[:,locr]

    new_cases = []
    for i in range(len(bosu)):
        if i == 0:
            new_cases.append(0)
        else:
            diff = bosu.iloc[i] - bosu.iloc[i-1]
            new_cases.append(diff)
            
    return new_cases
"""