import pandas as pd
import numpy as np
import os
import glob
import re
import datetime as dt
import requests
from bs4 import BeautifulSoup
import json

cwd = os.getcwd()
par = os.path.dirname(cwd)
parpar = os.path.dirname(par)

def case_extrapolate():
    # path to John Hopkins dataset
    johns_path = os.path.join(parpar, 'johns_hopkins', 'csse_covid_19_data', 'csse_covid_19_daily_reports', '')

    # list of all daily files with case records
    fnames = sorted(glob.glob(johns_path+'*.csv'))
    
    # first make a dictionary with all relevant countries (to later turn into dataframe)
    recent = pd.read_csv(johns_path + "01-01-2022.csv")
    countries = recent.Country_Region.value_counts().index
    country_dict = {c : [] for c in countries}
    
    # Now go through all relevant files and add cases to the dictionary
    ind = []
    for f in range(len(fnames)):
        date = re.search(r'\d{2}-\d{2}-\d{4}',fnames[f]).group(0)

        if dt.datetime.strptime(date, "%m-%d-%Y") >= dt.datetime(2020, 4, 12):
            ind.append(date)
            date_frame = pd.read_csv(fnames[f])
            
            for c in country_dict.keys():
                c_case = date_frame.loc[date_frame.Country_Region == c, "Confirmed"]
                
                if not c_case.empty:
                    country_dict[c].append(c_case.sum())
                else:
                    country_dict[c].append(np.nan)
                    
    country_cases = pd.DataFrame.from_dict(country_dict)
    country_cases.index = ind
    
    # re index so that dates are in order
    country_cases.index = pd.to_datetime(country_cases.index)
    country_cases = country_cases.sort_index()
    
    return country_cases



def get_abbreviations():
    url = "https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    plist = soup.find_all("div", class_="plainlist")[0]
    lis = plist.find_all("li")
    
    ab_dict = {}
    for l in lis:
        clist = l.text.replace(",", "").replace("\xa0\xa0", ",").split(",")
        ab_dict[clist[1]] = clist[0]
    
    # Manual Insertion of Kosovo
    ab_dict["Kosovo"] = "KOS"
      
    abbs = pd.DataFrame.from_dict(ab_dict, orient = "index")
    
    return abbs


def match_codes(iso3, world):
    #iso3 = pd.read_csv(os.path.join(par, "collected_data/country_codes.csv"), index_col = 0)        
    #world = pd.read_csv(os.path.join(par, "collected_data/world_dataset.csv"), index_col = 0) 
    
    translate = {}
    unk = []
    for c in world.columns:
        try:
            f = iso3.loc[c]
            translate[c] = c
        except:
            unk.append(c)
            
    manual = {"United States of America": "US",
              "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
              "Moldova Republic of": "Moldova", 
              "Russian Federation": "Russia",
              "Congo": "Congo (Brazzaville)", 
              "Congo Democratic Republic of the": "Congo (Kinshasa)",
              "Viet Nam" : "Vietnam",
              "Venezuela (Bolivarian Republic of)": "Venezuela",
              "Bolivia (Plurinational State of)": "Bolivia",
              "Côte d'Ivoire": "Cote d'Ivoire",
              "Taiwan Province of China": "Taiwan*",
              "Lao People's Democratic Republic": "Laos",
              "Myanmar": "Burma",
              "Syrian Arab Republic": "Syria",
              "Iran (Islamic Republic of)": "Iran",
              "Micronesia (Federated States of)": "Micronesia",
              "Korea Republic of" : "Korea, South",
              "Brunei Darussalam": "Brunei",
              "Tanzania United Republic of": "Tanzania"
              }
    
    
    for v in manual.values():
        try:
            unk.remove(v)
        except:
            pass
    
    for k in manual.keys():
        translate[k] = manual[k]
        
        
    h_names = []
    for i in iso3.index:
        try:
            h_names.append(translate[i])
        except:
            h_names.append(np.nan)
            
    iso3["h_name"] = h_names
    iso3.columns = ["abbrev", "h_name"]
    
    return iso3
    
    

def match_data(iso3, world):
    #iso3 = pd.read_csv(os.path.join(par, "collected_data/country_codes.csv"), index_col = 0)        
    #world = pd.read_csv(os.path.join(par, "collected_data/world_dataset.csv"), index_col = 0) 
    
    rename = []
    for c in world.columns:
        try:
            ab = iso3.loc[iso3.h_name == c]
            rename.append(ab.abbrev[0])
        except:
            world = world.drop(c, axis = 1)
        
    world.columns = rename
    
    return world
    
    
def alpha2(iso3):
    adj = pd.read_csv(os.path.join(par, "collected_data/country_adjacency.csv"))
    
    # translate from wikipedia iso3 list to geographical alpha2 list
    translate = {"Åland Islands": "Aland Islands",
                 "Bonaire Sint Eustatius and Saba": "Bonaire, Sint Eustatius and Saba",
                 "Saint Barthélemy": "Saint Barthelemy",
                 "Bolivia (Plurinational State of)": "Bolivia (Plurinational State Of)",
                 "Côte d'Ivoire": "Cote d’Ivoire",
                 "Congo Democratic Republic of the": "Congo (the Democratic Republic of the)",
                 "Curaçao": "Curacao",
                 "Gambia": "Gambia (the)",
                 "Korea Republic of": "Korea (the Republic of)",
                 "Moldova Republic of": "Moldova (the Republic of)",
                 "Palestine State of": "Palestine, State of",
                 "Réunion": "Reunion",
                 "Saint Helena Ascension and Tristan da Cunha": "Saint Helena,\"Ascension and Tristan da Cunha",
                 "Taiwan Province of China": "Taiwan (Province of China)",
                 "Tanzania United Republic of": "Tanzania (the United Republic of)"}
    
    
    alpha2s = []
    for c in iso3.index:
        adjs = adj.loc[adj.country_name == c]
        if adjs.empty:
            adjs = adj.loc[adj.country_name == translate[c]]
            
            if adjs.empty:
                print(c)
                alpha2s.append(np.nan)
                continue
        
        alpha2s.append(adjs.country_code.to_list()[0])
        
    iso3["alpha2"] = alpha2s
    
    return iso3
                

        
    
def drop_irrelevant(world):
    #world = pd.read_csv(os.path.join(par, "collected_data/world_dataset.csv"), index_col = 0)
    
    world = world.dropna(axis = 1, how = "any")
    
    for c in world.columns:
        if world[c].max() < 100:
            world = world.drop(c, axis = 1)
            
    return world
        

def clean_adjacency(codes):
    adj = pd.read_csv(os.path.join(par, "collected_data/country_adjacency.csv"))
    
    country_codes = []
    for c in adj.country_code:
        if not pd.isnull(c):
            alpha3 = codes.loc[codes.alpha2 == c].abbrev.values[0]
            country_codes.append(alpha3)
        else:
            country_codes.append(np.nan)
    
    country_border_codes = []
    for c in adj.country_border_code:
        if not pd.isnull(c):
            alpha3 = codes.loc[codes.alpha2 == c].abbrev.values[0]
            country_border_codes.append(alpha3)
        else:
            country_border_codes.append(np.nan)
        
    adj["country_code"] = country_codes
    adj["country_border_code"] = country_border_codes
    
    adj.to_csv(os.path.join(par, "collected_data/country_adjacency.csv"))

def make_laplacian(world):
    adj = pd.read_csv(os.path.join(par, "collected_data/country_adjacency.csv"))
    
    laplacian = np.zeros((len(world.columns), len(world.columns)))
    indexer = {world.columns[i] : i for i in range(len(world.columns))}
    for c in range(len(world.columns)):
        adjs = adj.loc[adj.country_code == world.columns[c]]
        laplacian[c,c] = len(adjs)
        for a in adjs.country_border_code:
            try:
                ind = indexer[a]
                laplacian[c, ind] = -1
                laplacian[ind, c] = -1
            except:
                laplacian[c,c] -= 1
                
    laplacian = pd.DataFrame(laplacian, index = world.columns, columns = world.columns)

    laplacian.to_csv(os.path.join(par, "collected_data/worldLaplacian.csv"))
    return laplacian
    
    
def population(world):
    pop_frame = pd.read_csv(os.path.join(par, "collected_data/country_population_data.csv"), index_col = 0)
    
    manual_population = {"TWN" : 23570000, "ERI": 6081000}
    
    pops = []
    for c in world.columns:
        lookup = pop_frame.loc[pop_frame["Country Code"] == c]
        try:
            pops.append(manual_population[c])
        except:
            pops.append(lookup["2020"].values[0])
        
    clean_pop_frame = pd.DataFrame([world.columns, pops]).T
    clean_pop_frame.columns = ["Country", "Population"]
    print(clean_pop_frame)
    clean_pop_frame.to_csv(os.path.join(par, "collected_data/world_population.csv"))
    return clean_pop_frame
        
    
        
    
        
    

case_frame = case_extrapolate()
#abbrevs = get_abbreviations()
#abbrevs = match_codes(abbrevs, case_frame)
#abbrevs = alpha2(abbrevs)
#abbrevs.to_csv(os.path.join(par, "collected_data/country_codes.csv"))
#clean_adjacency(abbrevs)
abbrevs = pd.read_csv(os.path.join(par, "collected_data/country_codes.csv"), index_col = 0)
case_frame = match_data(abbrevs, case_frame)
case_frame = drop_irrelevant(case_frame)
case_frame.to_csv(os.path.join(par, "collected_data/world_dataset.csv"))
pop = population(case_frame)
lapl = make_laplacian(case_frame)

    

abbrevs = pd.read_csv(os.path.join(par, "collected_data/country_codes.csv"), index_col = 0)
case_frame = pd.read_csv(os.path.join(par, "collected_data/world_dataset.csv"), index_col = 0)




europe = ["Russian Federation", "Ukraine", "France", "Spain", "Sweden", "Norway", "Kazakhstan", "Germany",
          "Finland", "Poland", "Italy", "United Kingdom of Great Britain and Northern Ireland", "Romania", "Belarus", "Greece", "Bulgaria",
          "Hungary", "Portugal", "Austria", "Czechia", "Serbia", "Ireland", "Lithuania",
          "Latvia", "Croatia", "Bosnia and Herzegovina", "Slovakia", "Estonia", "Denmark", "Switzerland",
          "Netherlands", "Moldova Republic of", "Belgium", "Albania", "North Macedonia", "Turkey", "Slovenia", "Montenegro",
          "Kosovo", "Azerbaijan", "Luxembourg", "Georgia", "Andorra", "Malta", "Liechtenstein", "San Marino",
          "Monaco", "Cyprus", "Armenia"]

european_abbrevs = []

for country in europe:
    try:
        european_abbrevs.append(abbrevs.loc[country,"abbrev"])
    except:
        print(country)