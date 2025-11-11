#Assuming this is data with genders for all authors including those with initials for their first name.

import pandas as pd
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt

data_dir = '/Users/akashrathi/Documents/GithubClones/GenderDisparitiesPlasticSurgery/data'
figure_dir = '/Users/akashrathi/Documents/GithubClones/GenderDisparitiesPlasticSurgery/figures'
df = pd.read_csv(os.join(data_dir,'final_data.csv'))

data = df

data = data.drop(['level_0','index'], axis = 1) #OPTIONAL ****only use if there are indices that were previously reset.

# Limit data to papers with available authors
data = data[data['Author Count'] != 0]

data['MM_yearly_prop'] = 0
for i in range(0, len(data)):
    year = data['Publication Year'][i]-1995
    data['MM_yearly_prop'][i] = stacked_data.iloc[year,3]
    
data['MF_yearly_prop'] = 0
for i in range(0, len(data)):
    year = data['Publication Year'][i]-1995
    data['MF_yearly_prop'][i] = stacked_data.iloc[year,2]

data['FM_yearly_prop'] = 0
for i in range(0, len(data)):
    year = data['Publication Year'][i]-1995
    data['FM_yearly_prop'][i] = stacked_data.iloc[year,1]

data['FF_yearly_prop'] = 0
for i in range(0, len(data)):
    year = data['Publication Year'][i]-1995
    data['FF_yearly_prop'][i] = stacked_data.iloc[year,0]

#Dropping all authors with initials for their first names

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

#Total authorship overall
def fig1_overall(data, labels=['F/F', 'F/M', 'M/F', 'M/M'],colors=['green', 'orange', 'turquoise', 'navy']):
    # Calculate the percentage of each category for each year
    percentage_data = data.groupby(['Publication Year', 'Gender']).size().reset_index(name='Count')
    percentage_data['Percentage'] = percentage_data.groupby('Publication Year')['Count'].transform(lambda x: (x / x.sum()) * 100)

    # Pivot the data for a stacked area plot
    stacked_data = percentage_data.pivot(index='Publication Year', columns='Gender', values='Percentage').fillna(0)

    # Create a stacked area plot
    plt.figure(figsize=(12, 4))
    stacks = plt.stackplot(stacked_data.index, stacked_data['F/F'], stacked_data['F/M'], 
                           stacked_data['M/F'], stacked_data['M/M'], 
                           colors=colors, 
                           alpha=0.65, linewidth=0.5, edgecolor='black')

    # Set plot labels and title
    plt.title('Overall authorship breakdown by year')
    plt.xlabel('Year')
    plt.ylabel('Percentage')

    # Round x-axis to the nearest whole number and rotate the x-values to the left
    x_ticks = range(int(stacked_data.index.min()), int(stacked_data.index.max()) + 1, 5)
    plt.xticks(x_ticks, rotation=0, ha='left')
    #plt.xticks([round(x) for x in stacked_data.index], rotation=45, ha='left')

    # Remove tick marks on both axes
    plt.tick_params(axis='both', which='both', length=0)

    # Change background color to white
    plt.gca().set_facecolor('white')

    # Add labels to each section
    for stack, label in zip(stacks, labels):
        path = stack.get_paths()[0]
        x_center = path.vertices[:, 0].mean()
        y_center = path.vertices[:, 1].mean()
        plt.text(x_center, y_center, label, color='black', va='center', ha='left')


    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)


    # Remove the legend
    plt.legend().set_visible(False)

    # Add black borders to the entire plot
    plt.margins(0, 0)
    plt.savefig(os.join(figure_dir, 'authorship_overall.pdf'), format='pdf')

    # Show the plot
    plt.show()

#Calling the function
fig1_overall(data)

# Total authorship by journal name
def fig1_journal(data, colors=['green', 'orange', 'turquoise', 'navy']):
    # Iterate over unique journals
    for journal in data['Journal Name'].unique():
        # Filter data for the current journal
        journal_data = data[data['Journal Name'] == journal]

        # Calculate the percentage of each category for each year
        percentage_data = journal_data.groupby(['Publication Year', 'Gender']).size().reset_index(name='Count')
        percentage_data['Percentage'] = percentage_data.groupby('Publication Year')['Count'].transform(lambda x: (x / x.sum()) * 100)

        # Pivot the data for a stacked area plot
        stacked_data = percentage_data.pivot(index='Publication Year', columns='Gender', values='Percentage').fillna(0)

        # Create a stacked area plot
        plt.figure(figsize=(12, 4))
        stacks = plt.stackplot(stacked_data.index, stacked_data['F/F'], 
                               stacked_data['F/M'], stacked_data['M/F'], 
                               stacked_data['M/M'], colors=colors, 
                               alpha=0.7, linewidth=0.5, edgecolor='black')

        # Set plot labels and title
        plt.title(f'{journal}')
        plt.xlabel('Year')
        plt.ylabel('Percentage')

        # Round x-axis to the nearest whole number and rotate the x-values to the left
        x_ticks = range(int(stacked_data.index.min()), int(stacked_data.index.max()) + 1, 5)
        plt.xticks(x_ticks, rotation=0, ha='left')
        #plt.xticks([round(x) for x in stacked_data.index], rotation=45, ha='left')

        # Remove tick marks on both axes
        plt.tick_params(axis='both', which='both', length=0)

        # Change background color to white
        plt.gca().set_facecolor('white')

        # Add labels to each section
        #for stack, label in zip(stacks, ['F/F', 'F/M', 'M/F', 'M/M']):
         #   path = stack.get_paths()[0]
            #x_center = path.vertices[:, 0].quartile(0.75)
            #y_center = path.vertices[:, 1].mean()
          #  plt.text(x_center, y_center, label, color='black', va='bottom', ha='left')

        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)

        # Remove the legend
        plt.legend().set_visible(False)

        # Add black borders to the entire plot
        plt.margins(0, 0)

        plt.savefig(os.join(figure_dir, f'{journal}_authorship.pdf'), format='pdf')

        # Show the plot
        plt.show()

#Call the function
fig1_journal(data)

percentage_data = data.groupby(['Publication Year', 'Gender']).size().reset_index(name='Count')
percentage_data['Percentage'] = percentage_data.groupby('Publication Year')['Count'].transform(lambda x: (x / x.sum()))
stacked_data = percentage_data.pivot(index='Publication Year', columns='Gender', values='Percentage').fillna(0)

