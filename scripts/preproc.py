import os
import pandas as pd
import numpy as np
import statistics
import math

#Load data
data_dir = '/Users/akashrathi/Documents/GithubClones/GenderDisparitiesPlasticSurgery/data'
cleaned = "Glane.csv"

file_path = os.path.join(data_dir, cleaned)
df = pd.read_csv(file_path)
data = df
data = data.drop(['Unnamed: 0'], axis = 1)

# Drop unknown author names
drop_list = []

for i,name in enumerate(data['label_first_author']):
  if name == 'unknown':
    drop_list.append(i)
data = data.drop(drop_list, axis=0).reset_index()

drop_list = []

for i,name in enumerate(data['label_last_author']):
  if name == 'unknown':
    drop_list.append(i)
data = data.drop(drop_list, axis=0).reset_index()

# New column with total authors per paper
data['Author Count'] = 0
for i in range(0,data.shape[0]):
   data['Author Count'][i] = len(data['Author Full Names'][i].split(';'))

# Create dummy var for review papers
value_to_dummy = 'Review'

# Create a dummy variable for the specified value
dummy_variable = pd.get_dummies(data['Article Type'] == value_to_dummy, prefix=f'{value_to_dummy}_dummy', drop_first=True)

# Concatenate the dummy variable with the original DataFrame
data = pd.concat([data, dummy_variable], axis=1)

# Label first/last author gender combination (M/M, F/M, M/F, F/F)
data['Gender'] = ' '

for i, author1 in enumerate(data['label_first_author']):
    if data['label_first_author'][i] == 'male' and data['label_last_author'][i] == 'male':
        data.at[i, 'Gender'] = 'M/M'
    elif data['label_first_author'][i] == 'male' and data['label_last_author'][i] == 'female':
        data.at[i, 'Gender'] = 'M/F'
    elif data['label_first_author'][i] == 'female' and data['label_last_author'][i] == 'male':
        data.at[i, 'Gender'] = 'F/M'
    elif data['label_first_author'][i] == 'female' and data['label_last_author'][i] == 'female':
        data.at[i, 'Gender'] = 'F/F'

data.Gender.value_counts()

# Count total first/last author citations for each paper
data['total_author_cites'] = 0
for i in range(0,data.shape[0]):
  a1_name = data.loc[i,'first_author']
  a2_name = data.loc[i,'last_author']
  a1_name_count = data.loc[data['first_author']==a1_name].shape[0]
  a2_name_count = data.loc[data['last_author']==a2_name].shape[0]
  total_count = a1_name_count + a2_name_count
  data['total_author_cites'][i]= total_count

# Count total number of other gender combinations cited per paper (i.e. how many F/M papers cited by a M/M paper)
data['M/M'] = 0
data['M/F'] = 0
data['F/M'] = 0
data['F/F'] = 0
data['M/M Self Cite'] = 0
data['M/F Self Cite'] = 0
data['F/M Self Cite'] = 0
data['F/F Self Cite'] = 0


index = 0
doi = ' '
empty_doi = []
citation_count = 0
for i,cite in enumerate(data['Cited References'][:]):
  for j,doi_string in enumerate(cite.split()):
      if doi_string == 'doi':
        doi = cite.split()[j+1][0:-1]
        citation_count+=1
        if data[data['DOI'] == doi].shape[0] == 0:
          empty_doi.append(i)
        else:
          print(i)
          index = data[data['DOI'] == doi].index
          gender_value = data.loc[index, 'Gender'].iloc[0]
          a1_name = data.loc[index,'first_author']
          a2_name = data.loc[index,'last_author']
          a1_Name_ref = data.loc[i,'first_author']
          a2_Name_ref = data.loc[i,'last_author']
          if (a1_name != a1_Name_ref).any() and (a2_name != a2_Name_ref).any() and (a1_name != a2_Name_ref).any() and (a2_name != a2_Name_ref).any():
          #if (a1_forename != a1_forename_ref).any() and (a1_surname != a1_surname_ref).any() and (a2_forename != a2_forename_ref).any() and (a2_surname != a2_surname_ref).any() and (a1_forename != a2_forename_ref).any() and (a1_surname != a2_surname_ref).any() and (a2_forename !=a1_forename_ref).any() and (a2_surname != a1_surname_ref).any():
            if gender_value == 'M/M':
              data['M/M'][i] +=1
            elif gender_value == 'M/F':
              data['M/F'][i] +=1
            elif gender_value == 'F/M':
              data['F/M'][i] +=1
            elif gender_value == 'F/F':
              data['F/F'][i] +=1
          else:
            if gender_value == 'M/M':
              data['M/M Self Cite'][i] +=1
            elif gender_value == 'M/F':
              data['M/F Self Cite'][i] +=1
            elif gender_value == 'F/M':
              data['F/M Self Cite'][i] +=1
            elif gender_value == 'F/F':
              data['F/F Self Cite'][i] +=1

drop_list = []
for i,first in enumerate(data.first_author):
  first_name = first.split(',')[1]
  if len(first_name) <= 3:
    drop_list.append(i)

for i,first in enumerate(data.last_author):
  first_name = first.split(',')[1]
  if len(first_name) <= 3:
    drop_list.append(i)

data = data.drop(drop_list, axis=0).reset_index()

final_label = 'final_data.csv'
final_path = os.path.join(data_dir, final_label)
data.to_csv(final_path, index=False)