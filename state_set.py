import os
import glob
import re
import pandas as pd
import datetime as dt
import random
import numpy as np
import copy
from matrix_operation import mat_opr

# Quick class to import a combined version of the Johns Hopkins Case data
#
# Setting get_dat to True allows an update to the dataset
# setting saver to True allows saving of the updated dataset
class state_data(mat_opr):
    def __init__(self, get_dat = False, saver = False):
        if get_dat:
            self.state_df = self.get_data()

            if saver:
                self.state_df.to_csv("state_dataset.csv")

        else:
            try:
                # If this is being imported as a module
                cwd = os.path.dirname(os.path.realpath(__file__))
                combo = os.path.join(cwd, "state_dataset.csv")
                self.combined = pd.read_csv(combo, index_col = 0)
            except:
                # If being used in the original file location
                self.combined = pd.read_csv("state_dataset.csv", index_col = 0)

    def get_data(self):
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

        state_cases.pop('Recovered')
        state_cases.pop('American Samoa')
        state_cases.pop('Diamond Princess')
        state_cases.pop('Northern Mariana Islands')
        cumul = pd.DataFrame(data=state_cases, index=ind)
        return cumul

        