# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 18:49:17 2020

@author: juan.alric
"""
import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

del df["Unnamed: 0"]

df = df.copy()

# Salary parsing

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

df = df[df['Salary Estimate'] != '-1']

salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_k_d = salary.apply(lambda x: x.replace('K', '').replace('$', ''))

min_hr = minus_k_d.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))

df['avg_salary'] = (df.max_salary + df.min_salary) / 2

# Company name text only

df['company_text'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)

# State field

df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])

df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)


# Age of the company

df['age'] = df.Founded.apply(lambda x: x if x < 1 else 2020 - x)

# Parsing job description (python, r studio, spark, aws, excel)

print(df['Job Description'][0])

df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
print(df.python_yn.value_counts())

df['r_studio_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
print(df.r_studio_yn.value_counts())

df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark'  in x.lower() else 0)
print(df.spark_yn.value_counts())


df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws'  in x.lower() else 0)
print(df.aws_yn.value_counts())

df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel'  in x.lower() else 0)
print(df.excel_yn.value_counts())

df.to_csv('salary_data_cleaned.csv', index=False)

df2 = pd.read_csv('salary_data_cleaned.csv')