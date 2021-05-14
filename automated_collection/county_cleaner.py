import os
import glob
import re
import pandas as pd
import datetime as dt
import random
import numpy as np
import copy


# a few functions related to cleaning the county level data

def county_cases(saver = False):
    # this is a function which can be used to get a nice version of county case data

    # this will update the data If you do a git pull from Johns hopkins beforehand
    # and setting saver to true will save the updated dataset.

    # Unfortunately this takes quite a bit of time because it is alot of data to sort through
    # I might look into making it faster in the future
    cwd = os.getcwd()
    par = os.path.dirname(cwd)
    parpar = os.path.dirname(par)

    '''
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
    '''

    # path to John Hopkins dataset
    johns_path = os.path.join(parpar, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports', '')

    fnames = sorted(glob.glob(johns_path+'*.csv'))
    ind = []

    # a recent set to help me get consistent county names
    recent = pd.read_csv(johns_path + "04-12-2021.csv")
    us = recent.loc[recent.Country_Region == "US"]

    # some locations not being considered
    nons = ["Recovered", "Diamond Princess", "Grand Princess", "Northern Mariana Islands", "Guam", "Virgin Islands"]

    multi = []

    for state in us.Province_State.value_counts().index:
        if state in nons:
            pass
        else:
            state_loc = us.loc[us.Province_State == state]
            for county in state_loc.Admin2:
                fips = state_loc.loc[state_loc.Admin2 == county].FIPS 
                fips = fips[fips.index[0]]

                if not np.isnan(fips):
                    multi.append((state,county,int(fips)))

    locations = pd.MultiIndex.from_tuples(multi, names=["state", "county", "fips"])
    count_cases = [[] for i in range(len(locations))]

    tot = 0
    for f in range(len(fnames)):
        date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)

        if dt.datetime.strptime(date, "%m-%d-%Y") >= dt.datetime(2020, 4, 12):
            ind.append(date)
            date_frame = pd.read_csv(fnames[f])
            date_frame = date_frame.loc[date_frame.Country_Region == "US"]
            
            for d in range(len(locations)):
                try:
                    #stater = date_frame.loc[date_frame.Province_State == locations[d][0]]
                    #counter = stater.loc[stater.Admin2 == locations[d][1]]
                    counter = date_frame.loc[date_frame.FIPS == locations[d][2]]
                    count_cases[d].append(counter.loc[counter.index[0],"Confirmed"])
                except:
                    tot += 1
                    count_cases[d].append(np.nan)

        else:
            pass

    cumul = pd.DataFrame(data=count_cases, index=locations, columns = ind)

    days = []
    for inder in cumul.columns:
        days.append(dt.datetime.strptime(inder, "%m-%d-%Y"))
    
    cumul.columns = days
    cumul = cumul.reindex(sorted(days), axis = 1)

    # drop some more unneccesary columns
    h = cumul.index.get_level_values(1)
    uns = h[h.str.startswith("Out of")].tolist()
    to_drop = []
    for ind in cumul.index:
        if ind[1] in uns or ind[1] == 'Unassigned':
            to_drop.append(ind)

    cumul = cumul.drop(to_drop,axis=0)

    # fill NaN + only considering the timeframe from 4/12/20 - present
    cumul = cumul.fillna(0)

    if saver:
        cumul.to_csv(os.path.join(par, "collected_data/county_dataset.csv"))

    return cumul





def county_census(saver = False):
    # to clean up the census data for use
    cwd = os.getcwd()
    par = os.path.dirname(cwd)
    parpar = os.path.dirname(par)

    county_case = pd.read_csv(os.path.join(par, "collected_data/county_dataset.csv"), index_col = [0,1,2])
    #opener = pd.read_excel("collected_data/co-est2019-alldata.xlsx")
    opener = pd.read_csv(os.path.join(par, "collected_data/co-est2019-alldata.csv"), engine = 'python')
    puerto = pd.read_csv(os.path.join(par, "collected_data/puerto_rico_census.csv"), index_col = 0)

    rel = opener['POPESTIMATE2019']
    counts = []
    drops = []
    dt = 0
    for ind in opener.index:
        locat = opener.iloc[ind,:]
        state = locat.STNAME
        county = locat.CTYNAME
        fips = locat.STATE * 1000 + locat.COUNTY

        if state == county:
            if state == 'District of Columbia':
                if dt == 0:
                  counts.append((state, county, fips))
            else:  
                drops.append(ind)
        else:
            counts.append((state, county, fips))

    multer = pd.MultiIndex.from_tuples(counts, names = ["state", "county", "fips"])
    rel = rel.drop(drops, axis = 0)
    rel.index = multer
    rel = pd.DataFrame(rel)
    rel.columns = ['Population Estimate']

    municipios = []
    for mun in puerto.NAME:
        #mun = mun.replace(' Municipio', '')
        municipios.append(('Puerto Rico', mun, 72000 + int(puerto.loc[puerto.NAME == mun].MUNICIPIO)))

    pmult = pd.MultiIndex.from_tuples(municipios, names = ['state', 'county', 'fips'])

    puerto = puerto.loc[:, ['POPESTIMATE']]
    puerto.columns = ['Population Estimate']
    puerto.index = pmult 

    rel = pd.concat([rel, puerto])
    #rel.columns = ['Population Estimate']

    if saver:
        rel.to_csv(os.path.join(par, "collected_data/county_census.csv"))
    return rel



if __name__ == "__main__":
    county_cases()
    county_census()