# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:49:17 2020

@author: juan.alric
"""
import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

del df["Unnamed: 0"]

df = df.copy()