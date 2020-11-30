import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import datetime as dt
import os

driver = webdriver.Chrome()
driver.get("https://www.nytimes.com/interactive/2020/us/covid-college-cases-tracker.html?referringSource=articleShare")
html = driver.page_source

soup = BeautifulSoup(html, "html.parser")
#school_list = soup.find("div", id = "schoolList")
school_list = soup.select('body > div > main > article > section > div > div.g-story.g-freebird.g-max-limit ')
school_list = school_list[0].select('#searchlist > div > .list_container > #schoolList')
state = list(school_list[0].children)

table = []
for i in range(len(state)):
    #s_name = state[i].find_all("div", class_= "statename")
    s_name = state[i].select('div > div > .statename')
    name = [s_name[0].get_text(strip = True)]
    #state_schools = state[i].find_all("div", class_="schoolcard individualschools")
    state_schools = state[i].select('div > .schoolholder > div')


    for j in state_schools:
        ind_school = [text for text in j.stripped_strings]
        if ind_school != []:
            ind_school += name
            table.append(ind_school)

frame = pd.DataFrame(table)
frame = frame.drop(frame.iloc[:,[3,4]], axis = 1)
frame.columns = ["School", "Cases", "City", "State"]
today = dt.date.today()
this = today.strftime("%m_%d_%y")
today = today.strftime("%m-%d-%Y")

frame["Date"] = today
print(frame)

curr_d = os.getcwd()
file_name = os.path.join(curr_d, "UniversityCases","university_cases_" + this + ".csv")
frame.to_csv(file_name)

f = open('last_scrape.txt', 'a')
f.write(file_name + '\n')
f.close()


driver.quit()
