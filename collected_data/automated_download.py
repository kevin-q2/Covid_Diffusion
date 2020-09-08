import os
import pandas as pd
import numpy as np
import datetime as dt

#Downloads Covid data from various sources

#Johns Hopkins data downloaded from Git Hub
Johns_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/'
data_by_state = pd.read_csv(Johns_url + "04-12-2020" + ".csv")

#cdc data on deaths
"""
deaths_url = 'https://data.cdc.gov/api/views/r8kw-7aab/rows.csv'
death_data = pd.read_csv(deaths_url)
"""

#general data from the covid tracking project
covid_tracker_url = "https://covidtracking.com/api/v1/states/daily.csv"
covid_tracker_daily = pd.read_csv(covid_tracker_url, index_col = "date", parse_dates=True)

#NYC data
nyc_data_url = "https://raw.githubusercontent.com/nychealth/coronavirus-data/master/case-hosp-death.csv"
#nyc_data = pd.read_csv(nyc_data_url)

#worldwide data
world_url = "https://www.ecdc.europa.eu/sites/default/files/documents/daily_national_incidence_2020-08-13_174508_0.xlsx"
world_xls = pd.read_excel(world_url)

#Automatically updates to get the most current data
start = dt.date(2020, 4, 13)
end = dt.date.today()
delta = dt.timedelta(days=1)

while start <= end:
    date_string = start.strftime("%m-%d-%Y")
    try:
        new_day = pd.read_csv(Johns_url + date_string + ".csv")
        data_by_state = data_by_state.append(new_day)
    except:
        break
    start += delta

#saves to local file path provided
data_by_state.to_csv(r"C:\Users\kq146\code\Covid_data\collected_data\State_Data.csv", index = False)
#death_data.to_csv(r"C:\Users\kq146\code\Covid_data\collected_data\cdc_death_data.csv", index = False)
covid_tracker_daily.to_csv(r"C:\Users\kq146\code\Covid_data\collected_data\covid_tracker_daily.csv")
nyc_data.to_csv(r"C:\Users\kq146\code\Covid_data\collected_data\daily_nyc.csv")
world_xls.to_csv(r"C:\Users\kq146\code\Covid_data\collected_data\daily_worldwide.csv", encoding = "utf-8")
