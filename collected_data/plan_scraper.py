import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
import datetime as dt


driver = webdriver.Chrome()
driver.get("https://collegecrisis.shinyapps.io/dashboard/")
html = driver.page_source

soup = BeautifulSoup(html, "html.parser")
map = soup.find_all("div", class_="leaflet-popup-content")
print(map)
driver.quit()
