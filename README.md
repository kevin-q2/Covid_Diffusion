# Covid_Project
Data mining project to make predictions about COVID-19  

The goal is use the Robust Synthetic Control method outlined in this blog post:  
http://peerunreviewed.blogspot.com/2019/11/a-short-tutorial-on-robust-synthetic.html

Right now synthetic_control_tests demonstrates a few tests with the method, mostly for my own understanding


### Collected Data:

State_Data.csv contains general, state level data about Covid (Confirmed cases, deaths, hospitilizations, etc.) collected daily by John's Hopkins  
https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data  (Using U.S. daily state report)  
This data is updated everyday  

covid_tracker_daily.csv is downloaded from the Covid Tracking Project and updated daily with data from all US states  
It is similar to the John's Hopkins data but is a bit more detailed (includes more info test results, seperate columns for confirmed and probable deaths, etc.) 
About: https://covidtracking.com/   
https://covidtracking.com/data

daily_nyc.csv provides daily case numbers specifically for New York City. I used these numbers to help test the method in synthetic_control_tests
https://github.com/nychealth/coronavirus-data

daily_worldwide.csv has case data from around the world provided by the European CDC. This is another source that I use for my tests
https://www.ecdc.europa.eu/en/publications-data/data-national-14-day-notification-rate-covid-19
