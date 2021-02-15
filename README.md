# Covid_Project
Data mining project to make predictions about COVID-19 in Universities across America


### UniversityCases/ 
contains university case data scraped from the New York Times Dashboard (https://www.nytimes.com/interactive/2020/us/covid-college-cases-tracker.html?referringSource=articleShare) 

It is combined with case data from surrounding counties provided by Johns Hopkins (https://github.com/CSSEGISandData/COVID-19) with the help of this US cities database (https://github.com/kelvins/US-Cities-Database)

### matrix_operation.py, Dataset.py, and state_set.py 
are modules I made to create, update, and analyze the data in these files:
  ##### combined_dataset.csv:
  contains a compiled set of the New York Times university case data combined with case data on each of the Big 10 Schools collected here: 
  https://91-divoc.com/pages/covid-19-at-big-ten-conference-schools/

  ##### state_dataset.csv
  contains a compiled set of state confirmed case data collected from Johns Hopkins

### Analysis:

All current tests and analyses are made in jupyter notebooks in the analysis/ folder

### other_data/
has collected information about school enrollment, county populations, and school reopening plans.


