# Covid_Project
Data mining project to analyze the spread of COVID-19 across the US. Much of the work so far has focused on using NMF to deconstruct cumulative case counts into different patterns or waves. Graphs with more detail can be found within the files in the analysis/ folder.

### collected_data/ 
contains most of the data I am currently using. It includes clean sets of case data originally collected from Johns Hopkins: https://github.com/CSSEGISandData/COVID-19
Along with some collected census data for the purpose of normalization. 

### University Cases
We've also done some research into how COVID has affected universities across the country. 

UniversityCases/ contains university case data scraped from the New York Times Dashboard (https://www.nytimes.com/interactive/2020/us/covid-college-cases-tracker.html?referringSource=articleShare) 

It is combined with case data from surrounding counties provided by Johns Hopkins (https://github.com/CSSEGISandData/COVID-19) with the help of this US cities database (https://github.com/kelvins/US-Cities-Database)


### matrix_operation.py, Dataset.py, and state_set.py 
are modules I made to create, update, and analyze all of the data. I am currently working on getting rid of Dataset.py and state_set.py so that the code from the notebooks can be run universally. 

### Analysis:

All current tests and analyses are made in jupyter notebooks in the analysis/ folder


