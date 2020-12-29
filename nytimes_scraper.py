import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import datetime as dt
import os
import codecs
from selenium.webdriver.chrome.options import Options

# I was previously using selenium with chromedriver to get the webpage
# But to do it on the csa machines I'm now just downloading html files everyday with wget

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--diable-dev-shm-usage')
driver = webdriver.Chrome('/usr/bin/chromedriver',options=chrome_options)
driver.get("https://www.nytimes.com/interactive/2020/us/covid-college-cases-tracker.html?referringSource=articleShare")
html = driver.page_source

#h = codecs.open('nyt_scrape.html', 'r', 'utf-8')
soup = BeautifulSoup(html, "html.parser")
school_list = soup.select('body > div > main > article > section > div > div.g-story.g-freebird.g-max-limit ')
school_list = school_list[0].select('#searchlist > div > .list_container > #schoolList')
ranger = list(school_list[0].children)

table = []
for i in range(len(ranger)):
    num = 'div.list_statename.state' + str(i)
    s_name = soup.select('body > div > main > article > section > div > div.g-story.g-freebird.g-max-limit > #searchlist > div > .list_container > #schoolList > ' + num + ' > div > .statename')
    name = [s_name[0].get_text(strip = True)]

    state_schools = soup.select('body > div > main > article > section > div > div.g-story.g-freebird.g-max-limit > #searchlist > div > .list_container > #schoolList > ' + num + ' > .schoolholder')
    state_schools = state_schools[0].find_all("div", class_="schoolcard individualschools")

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
