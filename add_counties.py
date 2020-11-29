import pandas as pd
import numpy as np
import datetime as dt
import os

def add_the_cases(univ, county):
    univ_counties = univ.County
    county = county.loc[county.Country_Region == "US"]
    active = []
    confirmed = []
    for i in univ_counties:
        try:
            row = county.loc[county.Admin2 == i]
            case = row.Active.iloc[0]
            total = row.Confirmed.iloc[0]
            active.append(case)
            confirmed.append(total)
        except:
            active.append(None)
            confirmed.append(None)

    act_cases = pd.Series(active)
    tot_cases = pd.Series(confirmed)
    cases = pd.concat([act_cases, tot_cases], axis = 1)
    cases.columns = ["County_Active_Cases", "County_Total_Cases"]
    return cases

cities = pd.read_csv(r"C:\Users\kq146\code\covid_college_tracker\US-Cities-Database\csv\us_cities.csv")
#county_frame = pd.read_csv(r"C:\Users\kq146\code\Covid_data\collected_data\county_data.csv")
start = dt.datetime.today() - dt.timedelta(days=3)
end = dt.datetime.today()
delta = dt.timedelta(days=1)

f = open('last_scrape.txt', 'r')
files = [i.strip() for i in f.readlines()]

d = open('last_scrape.txt', 'w')
d.close()

for h in files:
    date_string = dt.datetime.strptime(h[-12:-4], "%m_%d_%y")
    under_scores = date_string.strftime("%m_%d_%y")
    other_date = date_string.strftime("%m-%d-%Y")

    university = pd.read_csv(h)
    county_file = r"C:\Users\kq146\code\covid_college_tracker\johns_hopkins\csse_covid_19_data\csse_covid_19_daily_reports"
    county_file = os.path.join(county_file, other_date)

    try:
        county_frame = pd.read_csv(county_file + ".csv")
        university = university.loc[:, ["School", "Cases", "City", "State", "Date"]]
        #university.columns = ["School", "Cases", "City", "State", "Date"]
        u_loc = university.loc[:,["City", "State"]]
        states = u_loc.State.unique()

        counties = []
        for i in states:
            uni_city = u_loc.loc[u_loc.State == i]
            lookup = cities.loc[cities.STATE_NAME == i]

            for j in uni_city.City:
                location = lookup.loc[lookup.CITY == j]
                try:
                    counties.append(location.COUNTY.iloc[0])
                except:
                    counties.append(None)
            county_series = pd.Series(counties)
            county_series.columns = ["County"]
            combined = pd.concat([university, county_series], axis = 1, ignore_index = True)
            combined.columns = ["School", "Cases", "City", "State", "Date", "County"]
            combined = combined[["School", "Cases", "City", "County", "State", "Date"]]

            count_cases = add_the_cases(combined, county_frame)
            with_cases = pd.concat([combined, count_cases], axis = 1)
            #with_cases.columns = ["School", "Cases", "City", "County", "State", "Date", "County_Cases"]
            with_cases = with_cases[["School", "Cases", "County_Active_Cases","County_Total_Cases", "City", "County", "State", "Date"]]
            #print(with_cases)
            with_cases.to_csv(r"C:\Users\kq146\code\covid_college_tracker\Covid_data\UniversityCases\university_cases_" + under_scores + ".csv")

    except FileNotFoundError:
        re = open('last_scrape.txt', 'a')
        re.write(h)
        re.close()
