# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:47:30 2020

@author: juan.alric
"""

import glassdoor_scraper as gs
import pandas as pd

path = "C:/Users/juan.alric/Desktop/ds_salary_project/chromedriver"

df = gs.get_jobs('data scientist', 1000, False, path, 10)

df.to_csv("glassdoor_jobs.csv",index=False)

df.head()
