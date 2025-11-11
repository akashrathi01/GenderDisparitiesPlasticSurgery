#Assuming this is data with genders for all authors including those with initials for their first name.

import pandas as pd
import numpy as np
import statistics
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pygam import LinearGAM, s, f, PoissonGAM
from tqdm import tqdm  # For progress bar

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


def gam_model(model_data, bootstrap_df, expected_df, bootstrap_name, expected_name, n_bootstrap=1000):
    # Define predictors and target
    test_data = model_data
    X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]

    # Use LabelEncoder for categorical variables
    label_encoder = LabelEncoder()
    X_test_encoded = X_test.copy()
    X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
    X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])

    y_test = test_data['M/M']/test_data['Author Count']  # Assuming 'M/M' is the category to predict
    
    
    
    train_data = data
    train_data = train_data[(train_data['Publication Year'].between(2006, 2023, inclusive=True))]
    X_train = train_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]

    # Use LabelEncoder for categorical variables
    label_encoder = LabelEncoder()
    X_train_encoded = X_train.copy()
    X_train_encoded['Journal Name'] = label_encoder.fit_transform(X_train['Journal Name'])
    X_train_encoded['Publication Month'] = label_encoder.fit_transform(X_train['Publication Month'])

    y_train = train_data['M/M']/train_data['Author Count']  # Assuming 'M/M' is the category to predict
    
    
    
    

    # Number of bootstrap iterations
    n_bootstrap = n_bootstrap

    lam = 0.01

    # Initialize list to store bootstrap results
    bootstrap_results = []


    ###DEFINE SAMPLES###
    n_samples = round(0.20*len(y))
    
    expected_proportions_bootstrap_list = []
    # Bootstrap loop
    for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
        # Sample with replacement
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
        X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
        y_train_bootstrap = y_train.iloc[sample_indices]

        # Fit the GAM model on the bootstrap sample
        gam_bootstrap = LinearGAM(s(0, n_splines=5) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

        # Predict the probabilities for each category
        expected_bootstrap = gam_bootstrap.predict(X_test_encoded)

        # Calculate over- and undercitation measures
        observed_proportions = (test_data['M/M']/test_data['Author Count']).mean()
        expected_proportions_bootstrap = expected_bootstrap.mean()

        delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


        # Append to bootstrap results
        bootstrap_results.append(delta_mean_percentage)
        expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

    bootstrap_df['M/M_bootstrap_vals'] = bootstrap_results
    expected_df['M/M_expected'] = expected_proportions_bootstrap_list

    # Calculate the 95% confidence interval
    se_means = np.std(bootstrap_results)
    mm_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
    # Print the confidence interval
    print(f'95% Confidence Interval: {mm_confidence_interval}')
    mm_p_val = calculate_p_value(bootstrap_results)
    print(f'p-value: {mm_p_val}')



    y_train = train_data['M/F']/train_data['Author Count']  # Assuming 'M/M' is the category to predict
    
    test_data = model_data
    X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]

    # Number of bootstrap iterations


    # Initialize list to store bootstrap results
    bootstrap_results = []

    expected_proportions_bootstrap_list = []

    # Bootstrap loop
    for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
        # Sample with replacement
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
        X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
        y_train_bootstrap = y_train.iloc[sample_indices]

        # Fit the GAM model on the bootstrap sample
        gam_bootstrap = LinearGAM(s(0, n_splines=5) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

        # Predict the probabilities for each category
        expected_bootstrap = gam_bootstrap.predict(X_test_encoded)

        # Calculate over- and undercitation measures
        observed_proportions = (test_data['M/F']/test_data['Author Count']).mean()
        expected_proportions_bootstrap = expected_bootstrap.mean()

        delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


        # Append to bootstrap results
        bootstrap_results.append(delta_mean_percentage)
        expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

    bootstrap_df['M/F_bootstrap_vals'] = bootstrap_results
    expected_df['M/F_expected'] = expected_proportions_bootstrap_list
    # Calculate the 95% confidence interval
    se_means = np.std(bootstrap_results)
    mf_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
    # Print the confidence interval
    print(f'95% Confidence Interval: {mf_confidence_interval}')
    mf_p_val = calculate_p_value(bootstrap_results)
    print(f'p-value: {mf_p_val}')





    y_train = train_data['F/M']/train_data['Author Count']  # Assuming 'M/M' is the category to predict
    
    test_data = model_data
    X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]


    # Initialize list to store bootstrap results
    bootstrap_results = []

    expected_proportions_bootstrap_list = []

    # Bootstrap loop
    for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
        # Sample with replacement
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
        X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
        y_train_bootstrap = y_train.iloc[sample_indices]

        # Fit the GAM model on the bootstrap sample
        gam_bootstrap = LinearGAM(s(0, n_splines=5) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

        # Predict the probabilities for each category
        expected_bootstrap = gam_bootstrap.predict(X_test_encoded)

        # Calculate over- and undercitation measures
        observed_proportions = (test_data['F/M']/test_data['Author Count']).mean()
        expected_proportions_bootstrap = expected_bootstrap.mean()

        delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions

        #lower_limit = -100  # Set your lower limit
        #upper_limit = 100  # Set your upper limit

        #delta_mean_percentage = np.clip(delta_mean_percentage, lower_limit, upper_limit)

        # Append to bootstrap results
        bootstrap_results.append(delta_mean_percentage)
        expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

    bootstrap_df['F/M_bootstrap_vals'] = bootstrap_results
    expected_df['F/M_expected'] = expected_proportions_bootstrap_list
    # Calculate the 95% confidence interval
    se_means = np.std(bootstrap_results)
    fm_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))

    # Print the confidence interval
    print(f'95% Confidence Interval: {fm_confidence_interval}')
    fm_p_val = calculate_p_value(bootstrap_results)
    print(f'p-value: {fm_p_val}')



    y_train = train_data['F/F']/train_data['Author Count']  # Assuming 'M/M' is the category to predict
    
    test_data = model_data
    X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]

    # Initialize list to store bootstrap results
    bootstrap_results = []
    
    expected_proportions_bootstrap_list = []

    # Bootstrap loop
    for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
        # Sample with replacement
        sample_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
        X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
        y_train_bootstrap = y_train.iloc[sample_indices]

        # Fit the GAM model on the bootstrap sample
        gam_bootstrap = LinearGAM(s(0, n_splines=5) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

        # Predict the probabilities for each category
        expected_bootstrap = gam_bootstrap.predict(X_test_encoded)

        # Calculate over- and undercitation measures
        observed_proportions = (test_data['F/F']/test_data['Author Count']).mean()
        expected_proportions_bootstrap = expected_bootstrap.mean()

        delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions

        # Append to bootstrap results
        bootstrap_results.append(delta_mean_percentage)
        expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

    bootstrap_df['F/F_bootstrap_vals'] = bootstrap_results
    expected_df['F/F_expected'] = expected_proportions_bootstrap_list
    # Calculate the 95% confidence interval
    se_means = np.std(bootstrap_results)
    ff_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))

    # Print the confidence interval
    print(f'95% Confidence Interval: {ff_confidence_interval}')
    ff_p_val = calculate_p_value(bootstrap_results)
    print(f'p-value: {ff_p_val}')
    
    #SAVE BOOTSTRAP_DF
    bootstrap_df.to_csv(os.join(figure_dir, f'{bootstrap_name}'), index=False)
    
    #SAVE EXPECTED_DF
    expected_df.to_csv(os.join(data_dir,f'{expected_name}'), index=False)
    
    return [mm_confidence_interval, mf_confidence_interval, fm_confidence_interval, ff_confidence_interval]

gam_model(model_data, bootstrap_df, expected_df, 'bootstrap_df_fig2_top', 'expected_df_fig2_top', 1000)


#Finding the placement of each bar. IT SHOULD BE NOTED THAT THE ACTUAL BOOTSTRAP VALUES ARE SOMETIMES UNRELIABLE.
# RUNNING THIS CODE WILL CALCULATE THE OVER/UNDERCITATION MANUALLY BASED ON EXPECTED AND OBSERVED VALUES

delta = bootstrap_df[['M/M_bootstrap_vals', 'M/F_bootstrap_vals',
                             'F/M_bootstrap_vals','F/F_bootstrap_vals']]
observed_df = model_data[['M/M', 'M/F','F/M','F/F']].reset_index(drop=True)
n_bootstrap = 1000
#Standard errors for each bar
se_means = np.std(delta)
conf_ints = 1.96*se_means/(n_bootstrap ** 0.5)

#You can also calculate 95% Confidence Intervals using this code which saves them to their own variables
# NOTE SE_MEANS IS USED IN THIS FIGURE DUE TO SMALLER VARIABILITY
mm_confidence_interval = conf_ints[0]
mf_confidence_interval = conf_ints[1]
fm_confidence_interval = conf_ints[2]
ff_confidence_interval = conf_ints[3]


def fig_plotted(fig_name, delta=delta,conf_ints=conf_ints,
                colors=['navy', 'turquoise', 'orange', 'green'], labels=['M/M', 'M/F', 'F/M', 'F/F']):
    # Define custom colors
    custom_colors = colors

    # Create a bar plot
    sns.barplot(x=labels, y=delta.mean(),
                palette=custom_colors,
                alpha=0.7,
                linewidth=0.7,
                edgecolor='black')

    # Add labels and title
    plt.ylabel('Over- and Undercitation (%)')
    plt.title('Overall Over-Under Citation Measures for Each Gender Pair')

    plt.grid(True, linestyle='--', alpha=0.7, zorder=0)


    # Remove the legend
    plt.legend().set_visible(False)

    # Plot error bars
    for i, (calc, interval) in enumerate(zip(delta.mean(), conf_ints)):
        plt.errorbar(i, calc, yerr=interval, color='black', fmt='none', linewidth=2, capsize=10)
    
    #OPTIONAL to limit the y-axis for formatting
    #plt.ylim(-8, 2)

    plt.savefig(os.join(figure_dir, f'{fig_name}'), format='pdf')

    # Show the plot
    plt.show()

fig_plotted('fig_2_top', delta, conf_ints)

# Figure 2 (bottom)

expected_df = pd.DataFrame(columns = ['M/M_expected', 'M/F_expected','F/M_expected','F/F_expected'])
bootstrap_df = pd.DataFrame(columns = ['M/M_bootstrap_vals', 'M/F_bootstrap_vals','F/M_bootstrap_vals','F/F_bootstrap_vals'])
model_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F']]

model_data = model_data[(model_data['Publication Year'] >= 2006) & (model_data['total_author_cites']>=11)]

n_bootstrap=1000

gam_model(model_data, bootstrap_df, expected_df, 'bootstrap_df_fig2_bottom', 'expected_df_fig2_bottom', n_bootstrap=n_bootstrap)

delta = bootstrap_df[['M/M_bootstrap_vals', 'M/F_bootstrap_vals',
                             'F/M_bootstrap_vals','F/F_bootstrap_vals']]
observed_df = model_data[['M/M', 'M/F','F/M','F/F']].reset_index(drop=True)


se_means = np.std(delta)
conf_ints = 1.96*se_means/(n_bootstrap ** 0.5)


mm_confidence_interval = conf_ints[0]
mf_confidence_interval = conf_ints[1]
fm_confidence_interval = conf_ints[2]
ff_confidence_interval = conf_ints[3]

fig_plotted('fig2_bottom',delta, conf_ints)


# Figure 3 (MM)

expected_df = pd.DataFrame(columns = ['M/M_expected', 'M/F_expected','F/M_expected','F/F_expected'])
bootstrap_df = pd.DataFrame(columns = ['M/M_bootstrap_vals', 'M/F_bootstrap_vals','F/M_bootstrap_vals','F/F_bootstrap_vals'])
model_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F']]

model_data = model_data[(model_data['Publication Year'] >= 2006) & (model_data['Gender']=='M/M')]

n_bootstrap = 1000

gam_model(model_data, bootstrap_df, expected_df, 'bootstrap_df_fig3_MM', 'expected_df_fig3_MM', n_bootstrap=n_bootstrap)

delta = bootstrap_df[['M/M_bootstrap_vals', 'M/F_bootstrap_vals',
                             'F/M_bootstrap_vals','F/F_bootstrap_vals']]


se_means = np.std(delta)
conf_ints = 1.96*se_means/(n_bootstrap ** 0.5)


mm_confidence_interval = conf_ints[0]
mf_confidence_interval = conf_ints[1]
fm_confidence_interval = conf_ints[2]
ff_confidence_interval = conf_ints[3]

fig_plotted('fig3_MM',delta,conf_ints)


# Figure 3 (Everyone Else)

expected_df = pd.DataFrame(columns = ['M/M_expected', 'M/F_expected','F/M_expected','F/F_expected'])
bootstrap_df = pd.DataFrame(columns = ['M/M_bootstrap_vals', 'M/F_bootstrap_vals','F/M_bootstrap_vals','F/F_bootstrap_vals'])
model_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F']]

model_data = model_data[(model_data['Publication Year'] >= 2006) & (model_data['Gender']!='M/M')]

n_bootstrap = 1000

gam_model(model_data, bootstrap_df, expected_df, 'bootstrap_df_fig3_everyone_else', 'expected_df_fig3_everyone_else', n_bootstrap=n_bootstrap)

delta = bootstrap_df[['M/M_bootstrap_vals', 'M/F_bootstrap_vals',
                             'F/M_bootstrap_vals','F/F_bootstrap_vals']]


se_means = np.std(delta)
conf_ints = 1.96*se_means/(n_bootstrap ** 0.5)


mm_confidence_interval = conf_ints[0]
mf_confidence_interval = conf_ints[1]
fm_confidence_interval = conf_ints[2]
ff_confidence_interval = conf_ints[3]


fig_plotted('fig3_everyone_else',delta, conf_ints)


# Figure 3 (MF)

expected_df = pd.DataFrame(columns = ['M/M_expected', 'M/F_expected','F/M_expected','F/F_expected'])
bootstrap_df = pd.DataFrame(columns = ['M/M_bootstrap_vals', 'M/F_bootstrap_vals','F/M_bootstrap_vals','F/F_bootstrap_vals'])
model_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F']]

model_data = model_data[(model_data['Publication Year'] >= 2006) & (model_data['Gender']=='M/F')]

n_bootstrap = 1000
gam_model(model_data, bootstrap_df, expected_df, 'bootstrap_df_fig3_MF', 'expected_df_fig3_MF', n_bootstrap=n_bootstrap)

delta = bootstrap_df[['M/M_bootstrap_vals', 'M/F_bootstrap_vals',
                             'F/M_bootstrap_vals','F/F_bootstrap_vals']]


se_means = np.std(delta)
conf_ints = 1.96*se_means/(n_bootstrap ** 0.5)


mm_confidence_interval = conf_ints[0]
mf_confidence_interval = conf_ints[1]
fm_confidence_interval = conf_ints[2]
ff_confidence_interval = conf_ints[3]

fig_plotted('fig3_MF',delta, conf_ints)


# Figure 3 (FM)

expected_df = pd.DataFrame(columns = ['M/M_expected', 'M/F_expected','F/M_expected','F/F_expected'])
bootstrap_df = pd.DataFrame(columns = ['M/M_bootstrap_vals', 'M/F_bootstrap_vals','F/M_bootstrap_vals','F/F_bootstrap_vals'])
model_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F']]

model_data = model_data[(model_data['Publication Year'] >= 2006) & (model_data['Gender']=='F/M')]

n_bootstrap=1000
gam_model(model_data, bootstrap_df, expected_df, 'bootstrap_df_fig3_FM', 'expected_df_fig3_FM', n_bootstrap=n_bootstrap)

delta = bootstrap_df[['M/M_bootstrap_vals', 'M/F_bootstrap_vals',
                             'F/M_bootstrap_vals','F/F_bootstrap_vals']]


se_means = np.std(delta)
conf_ints = 1.96*se_means/(n_bootstrap ** 0.5)


mm_confidence_interval = conf_ints[0]
mf_confidence_interval = conf_ints[1]
fm_confidence_interval = conf_ints[2]
ff_confidence_interval = conf_ints[3]

fig_plotted('fig3_FM',delta, conf_ints)


# Figure 3 (FF)

expected_df = pd.DataFrame(columns = ['M/M_expected', 'M/F_expected','F/M_expected','F/F_expected'])
bootstrap_df = pd.DataFrame(columns = ['M/M_bootstrap_vals', 'M/F_bootstrap_vals','F/M_bootstrap_vals','F/F_bootstrap_vals'])
model_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F']]

model_data = model_data[(model_data['Publication Year'] >= 2006) & (model_data['Gender']=='F/F')]

n_bootstrap=1000
gam_model(model_data, bootstrap_df, expected_df, 'bootstrap_df_fig3_FF', 'expected_df_fig3_FF', n_bootstrap=n_bootstrap)

delta = bootstrap_df[['M/M_bootstrap_vals', 'M/F_bootstrap_vals',
                             'F/M_bootstrap_vals','F/F_bootstrap_vals']]


se_means = np.std(delta)
conf_ints = 1.96*se_means/(n_bootstrap ** 0.5)


mm_confidence_interval = conf_ints[0]
mf_confidence_interval = conf_ints[1]
fm_confidence_interval = conf_ints[2]
ff_confidence_interval = conf_ints[3]

fig_plotted('fig3_FF',delta,conf_ints)