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


# Figure 4 (MM)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pygam import LinearGAM, s, f, PoissonGAM
from tqdm import tqdm  # For progress bar
import statistics

bootstrap_df = pd.DataFrame(columns = ['M/M_bootstrap_vals', 'M/F_bootstrap_vals','F/M_bootstrap_vals','F/F_bootstrap_vals'])
model_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F','MM_yearly_prop','MF_yearly_prop',
                  'FM_yearly_prop','FF_yearly_prop']]
model_data = model_data[(model_data['Publication Year'].between(2006, 2023, inclusive=True))]

mm_conf_ints = []
mf_conf_ints = []
fm_conf_ints = []
ff_conf_ints = []

mm_p_vals = []
mf_p_vals = []
fm_p_vals = []
ff_p_vals = []

mm_obs_p_values = []
mf_obs_p_values = []
fm_obs_p_values = []
ff_obs_p_values = []

mm_obs_ci = []
mf_obs_ci = []
fm_obs_ci = []
ff_obs_ci = []

def fig4_gam_model(model_data, bootstrap_df, bootstrap_df_name, start_year, end_year, n_bootstrap=1000):
    
    for year in range(start_year, end_year+1):
        print(year)
        train_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F','MM_yearly_prop','MF_yearly_prop',
                  'FM_yearly_prop','FF_yearly_prop']]
        
        train_data = train_data[(train_data['Publication Year'].between(2006, 2023, inclusive=True))]
                              
        X_train = train_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 
                            'Review_dummy_True']]
        label_encoder = LabelEncoder()
        X_train_encoded = X_train.copy()
        X_train_encoded['Journal Name'] = label_encoder.fit_transform(X_train['Journal Name'])
        X_train_encoded['Publication Month'] = label_encoder.fit_transform(X_train['Publication Month'])
        y_train = train_data['M/M']/train_data['Author Count']
        
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']=='M/M')]



        # Define predictors and target
        X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]


        # Use LabelEncoder for categorical variables
        label_encoder = LabelEncoder()
        X_test_encoded = X_test.copy()
        X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
        X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])


        # Number of bootstrap iterations
        n_bootstrap = n_bootstrap


        # Initialize list to store bootstrap results
        bootstrap_results = []
        observed = []
        expected = []

        ###DEFINE SAMPLES###
        n_samples = round(0.20*len(y_train))

        # Fit the GAM model on the bootstrap sample
        expected_proportions_bootstrap_list = []

        # Bootstrap loop
        for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
      
            # Sample with replacement
            X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
            X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
            y_train_bootstrap = y_train.iloc[sample_indices]
        
            gam_bootstrap = LinearGAM(s(0, n_splines=4) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

            # Predict the probabilities for each category
            probabilities_bootstrap = gam_bootstrap.predict(X_test_encoded)

            # Calculate over- and undercitation measures
            observed_proportions = (test_data['M/M']/test_data['Author Count']).mean()
            expected_proportions_bootstrap = probabilities_bootstrap.mean()

            delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


            # Append to bootstrap results
            bootstrap_results.append(delta_mean_percentage)
            expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)


        bootstrap_df[f'{year}_M/M_bootstrap_vals'] = bootstrap_results
        bootstrap_df[f'{year}_M/M_observed'] = observed_proportions
        bootstrap_df[f'{year}_M/M_expected'] = expected_proportions_bootstrap_list

        # Calculate the 95% confidence interval
        se_means = np.std(bootstrap_results)
        mm_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
        mm_conf_ints.append((mm_confidence_interval[1]-mm_confidence_interval[0])/2)  

        # Print the confidence interval
        print(f'95% Confidence Interval: {mm_confidence_interval}')
        mm_p_val = calculate_p_value(bootstrap_results)
        mm_p_vals.append(mm_p_val)
        print(f'p-value: {mm_p_val}')



        y_train = train_data['M/F']/train_data['Author Count']  # Assuming 'M/M' is the category to predict


        # Initialize list to store bootstrap results
        bootstrap_results = []
        observed = []
        expected = []
      
        
        
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']=='M/M')]

        # Define predictors and target
        X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]

        # Use LabelEncoder for categorical variables
        label_encoder = LabelEncoder()
        X_test_encoded = X_test.copy()
        X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
        X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])
        
        
        expected_proportions_bootstrap_list = []

        # Bootstrap loop
        for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
            X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
            y_train_bootstrap = y_train.iloc[sample_indices]
            gam_bootstrap = LinearGAM(s(0, n_splines=4) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

            # Predict the probabilities for each category
            probabilities_bootstrap = gam_bootstrap.predict(X_test_encoded)

            # Calculate over- and undercitation measures
            observed_proportions = (test_data['M/F']/test_data['Author Count']).mean()
            expected_proportions_bootstrap = probabilities_bootstrap.mean()

            delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


            # Append to bootstrap results
            bootstrap_results.append(delta_mean_percentage)
            expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

            # Append to bootstrap results
          
            observed.append(observed_proportions)
            expected.append(expected_proportions_bootstrap)
      
        bootstrap_df[f'{year}_M/F_bootstrap_vals'] = bootstrap_results
        bootstrap_df[f'{year}_M/F_observed'] = observed
        bootstrap_df[f'{year}_M/F_expected'] = expected_proportions_bootstrap_list

        # Calculate the 95% confidence interval
        se_means = np.std(bootstrap_results)
        mf_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
        mf_conf_ints.append((mf_confidence_interval[1]-mf_confidence_interval[0])/2) 

        # Print the confidence interval
        print(f'95% Confidence Interval: {mf_confidence_interval}')
        mf_p_val = calculate_p_value(bootstrap_results)
        mf_p_vals.append(mf_p_val)
        print(f'p-value: {mf_p_val}')





        y_train = train_data['F/M']/train_data['Author Count']  # Assuming 'M/M' is the category to predict
    
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']=='M/M')]

        # Define predictors and target
        X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]


        # Use LabelEncoder for categorical variables
        label_encoder = LabelEncoder()
        X_test_encoded = X_test.copy()
        X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
        X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])

        # Initialize list to store bootstrap results
        bootstrap_results = []
        observed = []
        expected = []
        expected_proportions_bootstrap_list = []
      


        # Bootstrap loop
        for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
         
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
            X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
            y_train_bootstrap = y_train.iloc[sample_indices]
            gam_bootstrap = LinearGAM(s(0, n_splines=4) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

            
            # Predict the probabilities for each category
            probabilities_bootstrap = gam_bootstrap.predict(X_test_encoded)

            # Calculate over- and undercitation measures
            observed_proportions = (test_data['F/M']/test_data['Author Count']).mean()
            expected_proportions_bootstrap = probabilities_bootstrap.mean()

            delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


            # Append to bootstrap results
            bootstrap_results.append(delta_mean_percentage)
            expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

            # Append to bootstrap results
            observed.append(observed_proportions)
            expected.append(expected_proportions_bootstrap)


        bootstrap_df[f'{year}_F/M_bootstrap_vals'] = bootstrap_results
        bootstrap_df[f'{year}_F/M_observed'] = observed
        bootstrap_df[f'{year}_F/M_expected'] = expected_proportions_bootstrap_list
    
    
        se_means = np.std(bootstrap_results)
        fm_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
        fm_conf_ints.append((fm_confidence_interval[1]-fm_confidence_interval[0])/2) 
        # Print the confidence interval
        print(f'95% Confidence Interval: {fm_confidence_interval}')
        fm_p_val = calculate_p_value(bootstrap_results)
        fm_p_vals.append(fm_p_val)
        print(f'p-value: {fm_p_val}')



      
        # Initialize list to store bootstrap results
        bootstrap_results = []
        observed = []
        expected = []
        expected_proportions_bootstrap_list = []
        
        y_train = train_data['F/F']/train_data['Author Count']
    
    
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']=='M/M')]

        # Define predictors and target
        X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]


        # Use LabelEncoder for categorical variables
        label_encoder = LabelEncoder()
        X_test_encoded = X_test.copy()
        X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
        X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])

        # Bootstrap loop
        for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
    
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
            X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
            y_train_bootstrap = y_train.iloc[sample_indices]
            gam_bootstrap = LinearGAM(s(0, n_splines=4) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)


            # Predict the probabilities for each category
            probabilities_bootstrap = gam_bootstrap.predict(X_test_encoded)

            # Calculate over- and undercitation measures
            observed_proportions = (test_data['F/F']/test_data['Author Count']).mean()
            expected_proportions_bootstrap = probabilities_bootstrap.mean()

            delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


            # Append to bootstrap results
            bootstrap_results.append(delta_mean_percentage)
            expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

            # Append to bootstrap results
            observed.append(observed_proportions)
            expected.append(expected_proportions_bootstrap)
            
        bootstrap_df[f'{year}_F/F_bootstrap_vals'] = bootstrap_results
        bootstrap_df[f'{year}_F/F_observed'] = observed
        bootstrap_df[f'{year}_F/F_expected'] = expected_proportions_bootstrap_list
    
        # Calculate the 95% confidence interval
        se_means = np.std(bootstrap_results)
        ff_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
        ff_conf_ints.append((ff_confidence_interval[1]-ff_confidence_interval[0])/2) 
        # Print the confidence interval
        print(f'95% Confidence Interval: {ff_confidence_interval}')
        ff_p_val = calculate_p_value(bootstrap_results)
        ff_p_vals.append(ff_p_val)
        print(f'p-value: {ff_p_val}')
    
    
      
        
        
        
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']=='M/M')]
        n_bootstraped = 1000
        observed_values = test_data['M/M']/test_data['Author Count']
        # Bootstrap resampling
        bootstrap_samples = np.random.choice(observed_values, size=(len(observed_values), n_bootstrap), replace=True)
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        # Calculate the 95% confidence interval
        mm_obs_ci.append(np.percentile(bootstrap_means, [2.5, 97.5]))
        mm_obs_p_val = calculate_p_value(bootstrap_samples)
        mm_obs_p_values.append(mm_obs_p_val)

        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']=='M/M')]
        n_bootstraped = 1000
        observed_values = test_data['M/F']/test_data['Author Count']
        # Bootstrap resampling
        bootstrap_samples = np.random.choice(observed_values, size=(len(observed_values), n_bootstrap), replace=True)
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        # Calculate the 95% confidence interval
        mf_obs_ci.append(np.percentile(bootstrap_means, [2.5, 97.5]))
        mf_obs_p_val = calculate_p_value(bootstrap_samples)
        mf_obs_p_values.append(mf_obs_p_val)



        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']=='M/M')]
        n_bootstraped = 1000
        observed_values = test_data['F/M']/test_data['Author Count']
        # Bootstrap resampling
        bootstrap_samples = np.random.choice(observed_values, size=(len(observed_values), n_bootstrap), replace=True)
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        # Calculate the 95% confidence interval
        fm_obs_ci.append(np.percentile(bootstrap_means, [2.5, 97.5]))
        fm_obs_p_val = calculate_p_value(bootstrap_samples)
        fm_obs_p_values.append(fm_obs_p_val)
        



        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']=='M/M')]
        n_bootstraped = 1000
        observed_values = test_data['F/F']/test_data['Author Count']
        # Bootstrap resampling
        bootstrap_samples = np.random.choice(observed_values, size=(len(observed_values), n_bootstraped), replace=True)
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        # Calculate the 95% confidence interval
        ff_obs_ci.append(np.percentile(bootstrap_means, [2.5, 97.5]))
        ff_obs_p_val = calculate_p_value(bootstrap_samples)
        ff_obs_p_values.append(ff_obs_p_val)
    
        bootstrap_df.to_csv(os.join(data_dir,f'{bootstrap_df_name}'), index=False)

fig4_gam_model(model_data, bootstrap_df, 'fig4_bootstrap_df', 2007, 2023, n_bootstrap=1000)


# Get bootstrap values        
MM_bootstrap_vals = []
for col in bootstrap_df.columns:
  if 'M/M_bootstrap' in col:
    MM_bootstrap_vals.append(bootstrap_df[col].mean())
MM_bootstrap_vals = [x for x in MM_bootstrap_vals if str(x) != 'nan']
#MM_errors = np.std(MM_bootstrap_vals, axis=0) / np.sqrt(len(MM_bootstrap_vals))


MF_bootstrap_vals = []
for col in bootstrap_df.columns:
  if 'M/F_bootstrap' in col:
    MF_bootstrap_vals.append(bootstrap_df[col].mean())
MF_bootstrap_vals = [x for x in MF_bootstrap_vals if str(x) != 'nan']
#MF_errors = np.std(MF_bootstrap_vals, axis=0) / np.sqrt(len(MF_bootstrap_vals))

FM_bootstrap_vals = []
for col in bootstrap_df.columns:
  if 'F/M_bootstrap' in col:
    FM_bootstrap_vals.append(bootstrap_df[col].mean())
FM_bootstrap_vals = [x for x in FM_bootstrap_vals if str(x) != 'nan']
#FM_errors = np.std(FM_bootstrap_vals, axis=0) / np.sqrt(len(FM_bootstrap_vals))

FF_bootstrap_vals = []
for col in bootstrap_df.columns:
  if 'F/F_bootstrap' in col:
    FF_bootstrap_vals.append(bootstrap_df[col].mean())
FF_bootstrap_vals = [x for x in FF_bootstrap_vals if str(x) != 'nan']
#FF_errors = np.std(FF_bootstrap_vals, axis=0) / np.sqrt(len(FF_bootstrap_vals))




# Calculate the expected values from GAM model
MM_expected_vals = []
for col in bootstrap_df.columns:
  if 'M/M_expected' in col:
    MM_expected_vals.append(bootstrap_df[col].mean())
MM_expected_vals = [x for x in MM_expected_vals if str(x) != 'nan']
#MM_errors = np.std(MM_bootstrap_vals, axis=0) / np.sqrt(len(MM_bootstrap_vals))


MF_expected_vals = []
for col in bootstrap_df.columns:
  if 'M/F_expected' in col:
    MF_expected_vals.append(bootstrap_df[col].mean())
MF_expected_vals = [x for x in MF_expected_vals if str(x) != 'nan']
#MF_errors = np.std(MF_bootstrap_vals, axis=0) / np.sqrt(len(MF_bootstrap_vals))

FM_expected_vals = []
for col in bootstrap_df.columns:
  if 'F/M_expected' in col:
    FM_expected_vals.append(bootstrap_df[col].mean())
FM_expected_vals = [x for x in FM_expected_vals if str(x) != 'nan']
#FM_errors = np.std(FM_bootstrap_vals, axis=0) / np.sqrt(len(FM_bootstrap_vals))

FF_expected_vals = []
for col in bootstrap_df.columns:
  if 'F/F_expected' in col:
    FF_expected_vals.append(bootstrap_df[col].mean())
FF_expected_vals = [x for x in FF_expected_vals if str(x) != 'nan']
#FF_errors = np.std(FF_bootstrap_vals, axis=0) / np.sqrt(len(FF_bootstrap_vals))



# Get observed values from data
MM_observed_vals = []
for col in bootstrap_df.columns:
  if 'M/M_observed' in col:
    MM_observed_vals.append(bootstrap_df[col].mean())
MM_observed_vals = [x for x in MM_observed_vals if str(x) != 'nan']
#MM_errors = np.std(MM_bootstrap_vals, axis=0) / np.sqrt(len(MM_bootstrap_vals))


MF_observed_vals = []
for col in bootstrap_df.columns:
  if 'M/F_observed' in col:
    MF_observed_vals.append(bootstrap_df[col].mean())
MF_observed_vals = [x for x in MF_observed_vals if str(x) != 'nan']
#MF_errors = np.std(MF_bootstrap_vals, axis=0) / np.sqrt(len(MF_bootstrap_vals))

FM_observed_vals = []
for col in bootstrap_df.columns:
  if 'F/M_observed' in col:
    FM_observed_vals.append(bootstrap_df[col].mean())
FM_observed_vals = [x for x in FM_observed_vals if str(x) != 'nan']
#FM_errors = np.std(FM_bootstrap_vals, axis=0) / np.sqrt(len(FM_bootstrap_vals))

FF_observed_vals = []
for col in bootstrap_df.columns:
  if 'F/F_observed' in col:
    FF_observed_vals.append(bootstrap_df[col].mean())
FF_observed_vals = [x for x in FF_observed_vals if str(x) != 'nan']


# MM expected vs. observed plot
mm_obs_ci_error = []
for i in mm_obs_ci:
    mm_obs_ci_error.append((np.diff(i)/2)[0])

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

rounded_years = np.ceil(years)
rounded_years = rounded_years.astype(int)

# Set the rounded values as tick labels
plt.xticks(years, rounded_years)
plt.xticks(np.arange(rounded_years.min(), rounded_years.max() + 1, 2))

plt.plot(years, MM_observed_vals, label='M/M', marker='o', color = 'navy')
plt.plot(years, MM_expected_vals, label='M/M', marker='o', color = 'navy', linestyle='--')

plt.errorbar(years, MM_observed_vals, yerr=np.mean(mm_obs_ci_error), label='M/M', marker='o', linestyle='', capsize=5, color = 'navy')
#plt.errorbar(years, MM_expected_vals, yerr=mm_errors, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')


# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.ylim()
plt.xlabel('Year')
plt.ylabel('Proportion cited')
plt.title('Citing authors: male and male (MM)')
plt.ylim(0.60,0.95)

plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_MM_MandM', format='pdf')

# Show the plot
plt.show()

print('observed slope CI:',(np.polyfit(years, MM_observed_vals, 1)[0]-np.mean(mm_obs_ci_error),
                           np.polyfit(years, MM_observed_vals, 1)[0]+np.mean(mm_obs_ci_error)))
print('observed slope p-value:',np.mean(mm_obs_p_values))
print('\n')
print('expected slope:',(np.polyfit(years, MM_expected_vals, 1)[0]))


# MF expected vs. observed plot
mf_obs_ci_error = []
for i in mf_obs_ci:
    mf_obs_ci_error.append((np.diff(i)/2)[0])

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

rounded_years = np.ceil(years)
rounded_years = rounded_years.astype(int)

# Set the rounded values as tick labels
plt.xticks(years, rounded_years)
plt.xticks(np.arange(rounded_years.min(), rounded_years.max() + 1, 2))

plt.plot(years, MF_observed_vals, label='M/M', marker='o', color = 'turquoise')
plt.plot(years, MF_expected_vals, label='M/M', marker='o', color = 'turquoise', linestyle='--')

plt.errorbar(years, MF_observed_vals, yerr=np.mean(mf_obs_ci_error), label='M/M', marker='o', linestyle='', capsize=5, color = 'turquoise')
#plt.errorbar(years, MF_exp_line, yerr=MF_errors, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')


# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.ylim(-1,1)
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: male and male (MM)')
plt.ylim(0,0.35)

plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_MM_MandF', format='pdf')

# Show the plot
plt.show()

print('observed slope CI:',(np.polyfit(years, MF_observed_vals, 1)[0]-np.mean(mf_obs_ci_error),
                           np.polyfit(years, MF_observed_vals, 1)[0]+np.mean(mf_obs_ci_error)))
print('observed slope p-value:',np.mean(mf_obs_p_values))
print('\n')
print('expected slope:',(np.polyfit(years, MF_expected_vals, 1)[0]))


# FM expected vs. observed plot

fm_obs_ci_error = []
for i in fm_obs_ci:
    fm_obs_ci_error.append((np.diff(i)/2)[0])

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

rounded_years = np.ceil(years)
rounded_years = rounded_years.astype(int)

# Set the rounded values as tick labels
plt.xticks(years, rounded_years)
plt.xticks(np.arange(rounded_years.min(), rounded_years.max() + 1, 2))

plt.plot(years, FM_observed_vals, label='M/M', marker='o', color = 'orange')
plt.plot(years, FM_expected_vals, label='M/M', marker='o', color = 'orange', linestyle='--')

plt.errorbar(years, FM_observed_vals, yerr=np.mean(fm_obs_ci_error), label='M/M', marker='o', linestyle='', capsize=5, color = 'orange')
#plt.errorbar(years, MF_exp_line, yerr=MF_errors, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')


# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.ylim(-1,1)
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: male and male (MM)')
plt.ylim(0,0.35)


plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_MM_FandM', format='pdf')

# Show the plot
plt.show()

print('observed slope CI:',(np.polyfit(years, FM_observed_vals, 1)[0]-np.mean(fm_obs_ci_error),
                           np.polyfit(years, FM_observed_vals, 1)[0]+np.mean(fm_obs_ci_error)))
print('observed slope p-value:',np.mean(fm_obs_p_values))
print('\n')
print('expected slope:',(np.polyfit(years, FM_expected_vals, 1)[0]))


# FF expected vs. observed plot

ff_obs_ci_error = []
for i in ff_obs_ci:
    ff_obs_ci_error.append((np.diff(i)/2)[0])

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

plt.plot(years, FF_observed_vals, label='M/M', marker='o', color = 'green')
plt.plot(years, FF_expected_vals, label='M/M', marker='o', color = 'green', linestyle='--')

plt.errorbar(years, FF_observed_vals, yerr=np.mean(ff_obs_ci_error), label='M/M', marker='o', linestyle='', capsize=5, color = 'green')
#plt.errorbar(years, MF_exp_line, yerr=MF_errors, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')


# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.ylim(-0.07,0.30)
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: male and male (MM)')
plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_MM_FandF', format='pdf')

# Show the plot
plt.show()

print('observed slope CI:',(np.polyfit(years, FF_observed_vals, 1)[0]-np.mean(ff_obs_ci_error),
                           np.polyfit(years, FF_observed_vals, 1)[0]+np.mean(ff_obs_ci_error)))
print('observed slope p-value:',np.mean(ff_obs_p_values))
print('\n')
print('expected slope:',(np.polyfit(years, FF_expected_vals, 1)[0]))

# Over- under-citation figure by MM authors

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

rounded_years = np.ceil(years)
rounded_years = rounded_years.astype(int)

# Set the rounded values as tick labels
plt.xticks(years, rounded_years)
plt.xticks(np.arange(rounded_years.min(), rounded_years.max() + 1, 2))

plt.plot(years, MM_bootstrap_vals, label='M/M', marker='o', color = 'navy')
plt.plot(years, MF_bootstrap_vals, label='M/F', marker='o', color = 'turquoise')
plt.plot(years, FM_bootstrap_vals, label='F/M', marker='o', color = 'orange')
plt.plot(years, FF_bootstrap_vals, label='F/F', marker='o', color = 'green')

plt.errorbar(years, MM_bootstrap_vals, yerr=np.mean(mm_conf_ints), label='M/M', marker='o', linestyle='', capsize=5, color = 'navy')
plt.errorbar(years, MF_bootstrap_vals, yerr=np.mean(mf_conf_ints), label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')
plt.errorbar(years, FM_bootstrap_vals, yerr=np.mean(fm_conf_ints), label='F/M', marker='o', linestyle='', capsize=5, color = 'orange')
plt.errorbar(years, FF_bootstrap_vals, yerr=np.mean(ff_conf_ints), label='F/F', marker='o', linestyle='', capsize=5, color = 'green')

# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: male and male (MM)')
#plt.ylim(-9,1)

plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_overall_ex', format='pdf')

# Show the plot
plt.show()

print('MM slope CI:',(np.polyfit(years, MM_bootstrap_vals, 1)[0]-np.mean(mm_conf_ints),
                           np.polyfit(years, MM_bootstrap_vals, 1)[0]+np.mean(mm_conf_ints)))
print('MM slope p-value:',np.mean(mm_p_vals))

print('MF slope CI:',(np.polyfit(years, MF_bootstrap_vals, 1)[0]-np.mean(mf_conf_ints),
                           np.polyfit(years, MF_bootstrap_vals, 1)[0]+np.mean(mf_conf_ints)))
print('MF slope p-value:',np.mean(mf_p_vals))
print('FM slope CI:',(np.polyfit(years, FM_bootstrap_vals, 1)[0]-np.mean(fm_conf_ints),
                           np.polyfit(years, FM_bootstrap_vals, 1)[0]+np.mean(fm_conf_ints)))
print('FM slope p-value:',np.mean(fm_p_vals))
print('FF slope CI:',(np.polyfit(years, FF_bootstrap_vals, 1)[0]-np.mean(ff_conf_ints),
                           np.polyfit(years, FF_bootstrap_vals, 1)[0]+np.mean(ff_conf_ints)))
print('FF slope p-value:',np.mean(ff_p_vals))

###############################################################################################

#GAM for FM, MF, FF authors

bootstrap_df = pd.DataFrame(columns = ['M/M_bootstrap_vals', 'M/F_bootstrap_vals','F/M_bootstrap_vals','F/F_bootstrap_vals'])
model_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F','MM_yearly_prop','MF_yearly_prop',
                  'FM_yearly_prop','FF_yearly_prop']]
model_data = model_data[(model_data['Publication Year'].between(2007, 2023, inclusive=True))]

mm_conf_ints = []
mf_conf_ints = []
fm_conf_ints = []
ff_conf_ints = []

mm_p_vals = []
mf_p_vals = []
fm_p_vals = []
ff_p_vals = []

mm_obs_p_values = []
mf_obs_p_values = []
fm_obs_p_values = []
ff_obs_p_values = []

mm_obs_ci = []
mf_obs_ci = []
fm_obs_ci = []
ff_obs_ci = []

def fig4_gam_model(model_data, bootstrap_df, bootstrap_df_name, start_year, end_year, n_bootstrap=1000):
    
    for year in range(start_year, end_year+1):
        print(year)
        train_data = data[['Publication Year', 'Publication Month',
                   'Author Count','Review_dummy_True','Gender',
                   'total_author_cites', 'Journal Name',
                   'Article Type','M/M','M/F','F/M','F/F','MM_yearly_prop','MF_yearly_prop',
                  'FM_yearly_prop','FF_yearly_prop']]
        
        train_data = train_data[(train_data['Publication Year'].between(2007, 2023, inclusive=True))]
                              
        X_train = train_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 
                            'Review_dummy_True']]
        label_encoder = LabelEncoder()
        X_train_encoded = X_train.copy()
        X_train_encoded['Journal Name'] = label_encoder.fit_transform(X_train['Journal Name'])
        X_train_encoded['Publication Month'] = label_encoder.fit_transform(X_train['Publication Month'])
        y_train = train_data['M/M']/train_data['Author Count']
        
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']!='M/M')]



        # Define predictors and target
        X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]


        # Use LabelEncoder for categorical variables
        label_encoder = LabelEncoder()
        X_test_encoded = X_test.copy()
        X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
        X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])


        # Number of bootstrap iterations
        n_bootstrap = n_bootstrap


        # Initialize list to store bootstrap results
        bootstrap_results = []
        observed = []
        expected = []

        ###DEFINE SAMPLES###
        n_samples = round(0.20*len(y_train))

        # Fit the GAM model on the bootstrap sample
        expected_proportions_bootstrap_list = []

        # Bootstrap loop
        for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
      
            # Sample with replacement
            X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
            X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
            y_train_bootstrap = y_train.iloc[sample_indices]
        
            gam_bootstrap = LinearGAM(s(0, n_splines=4) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

            # Predict the probabilities for each category
            probabilities_bootstrap = gam_bootstrap.predict(X_test_encoded)

            # Calculate over- and undercitation measures
            observed_proportions = (test_data['M/M']/test_data['Author Count']).mean()
            expected_proportions_bootstrap = probabilities_bootstrap.mean()

            delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


            # Append to bootstrap results
            bootstrap_results.append(delta_mean_percentage)
            expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)


        bootstrap_df[f'{year}_M/M_bootstrap_vals'] = bootstrap_results
        bootstrap_df[f'{year}_M/M_observed'] = observed_proportions
        bootstrap_df[f'{year}_M/M_expected'] = expected_proportions_bootstrap_list

        # Calculate the 95% confidence interval
        se_means = np.std(bootstrap_results)
        mm_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
        mm_conf_ints.append((mm_confidence_interval[1]-mm_confidence_interval[0])/2)  


        # Print the confidence interval
        print(f'95% Confidence Interval: {mm_confidence_interval}')
        mm_p_val = calculate_p_value(bootstrap_results)
        mm_p_vals.append(mm_p_val)
        print(f'p-value: {mm_p_val}')



        y_train = train_data['M/F']/train_data['Author Count']  # Assuming 'M/M' is the category to predict


        # Initialize list to store bootstrap results
        bootstrap_results = []
        observed = []
        expected = []
      
        
        
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']!='M/M')]

        # Define predictors and target
        X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]

        # Use LabelEncoder for categorical variables
        label_encoder = LabelEncoder()
        X_test_encoded = X_test.copy()
        X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
        X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])
        
        
        expected_proportions_bootstrap_list = []

        # Bootstrap loop
        for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
            X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
            y_train_bootstrap = y_train.iloc[sample_indices]
            gam_bootstrap = LinearGAM(s(0, n_splines=4) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

            # Predict the probabilities for each category
            probabilities_bootstrap = gam_bootstrap.predict(X_test_encoded)

            # Calculate over- and undercitation measures
            observed_proportions = (test_data['M/F']/test_data['Author Count']).mean()
            expected_proportions_bootstrap = probabilities_bootstrap.mean()

            delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


            # Append to bootstrap results
            bootstrap_results.append(delta_mean_percentage)
            expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

            # Append to bootstrap results
          
            observed.append(observed_proportions)
            expected.append(expected_proportions_bootstrap)
      
        bootstrap_df[f'{year}_M/F_bootstrap_vals'] = bootstrap_results
        bootstrap_df[f'{year}_M/F_observed'] = observed
        bootstrap_df[f'{year}_M/F_expected'] = expected_proportions_bootstrap_list

        # Calculate the 95% confidence interval
        se_means = np.std(bootstrap_results)
        mf_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
        mf_conf_ints.append((mf_confidence_interval[1]-mf_confidence_interval[0])/2) 

        # Print the confidence interval
        print(f'95% Confidence Interval: {mf_confidence_interval}')
        mf_p_val = calculate_p_value(bootstrap_results)
        mf_p_vals.append(mf_p_val)
        print(f'p-value: {mf_p_val}')





        y_train = train_data['F/M']/train_data['Author Count']  # Assuming 'M/M' is the category to predict
    
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']!='M/M')]

        # Define predictors and target
        X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]


        # Use LabelEncoder for categorical variables
        label_encoder = LabelEncoder()
        X_test_encoded = X_test.copy()
        X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
        X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])

        # Initialize list to store bootstrap results
        bootstrap_results = []
        observed = []
        expected = []
        expected_proportions_bootstrap_list = []
      


        # Bootstrap loop
        for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
         
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
            X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
            y_train_bootstrap = y_train.iloc[sample_indices]
            gam_bootstrap = LinearGAM(s(0, n_splines=4) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)

            
            # Predict the probabilities for each category
            probabilities_bootstrap = gam_bootstrap.predict(X_test_encoded)

            # Calculate over- and undercitation measures
            observed_proportions = (test_data['F/M']/test_data['Author Count']).mean()
            expected_proportions_bootstrap = probabilities_bootstrap.mean()

            delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


            # Append to bootstrap results
            bootstrap_results.append(delta_mean_percentage)
            expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

            # Append to bootstrap results
            observed.append(observed_proportions)
            expected.append(expected_proportions_bootstrap)


        bootstrap_df[f'{year}_F/M_bootstrap_vals'] = bootstrap_results
        bootstrap_df[f'{year}_F/M_observed'] = observed
        bootstrap_df[f'{year}_F/M_expected'] = expected_proportions_bootstrap_list
    
    
        se_means = np.std(bootstrap_results)
        fm_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
        fm_conf_ints.append((fm_confidence_interval[1]-fm_confidence_interval[0])/2) 
        
        # Print the confidence interval
        print(f'95% Confidence Interval: {mm_confidence_interval}')
        fm_p_val = calculate_p_value(bootstrap_results)
        fm_p_vals.append(fm_p_val)
        print(f'p-value: {fm_p_val}')



      
        # Initialize list to store bootstrap results
        bootstrap_results = []
        observed = []
        expected = []
        expected_proportions_bootstrap_list = []
        
        y_train = train_data['F/F']/train_data['Author Count']
    
    
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']!='M/M')]

        # Define predictors and target
        X_test = test_data[['Publication Year', 'Author Count','Publication Month','total_author_cites', 'Journal Name', 'Review_dummy_True']]


        # Use LabelEncoder for categorical variables
        label_encoder = LabelEncoder()
        X_test_encoded = X_test.copy()
        X_test_encoded['Journal Name'] = label_encoder.fit_transform(X_test['Journal Name'])
        X_test_encoded['Publication Month'] = label_encoder.fit_transform(X_test['Publication Month'])

        # Bootstrap loop
        for _ in tqdm(range(n_bootstrap), desc='Bootstrapping'):
    
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_train_encoded_reset_index = X_train_encoded.reset_index(drop=True)
            X_train_bootstrap = X_train_encoded_reset_index.iloc[sample_indices]
            y_train_bootstrap = y_train.iloc[sample_indices]
            gam_bootstrap = LinearGAM(s(0, n_splines=4) + s(1) + s(2) + s(3)+s(4)+s(5)).fit(X_train_bootstrap, y_train_bootstrap)


            # Predict the probabilities for each category
            probabilities_bootstrap = gam_bootstrap.predict(X_test_encoded)

            # Calculate over- and undercitation measures
            observed_proportions = (test_data['F/F']/test_data['Author Count']).mean()
            expected_proportions_bootstrap = probabilities_bootstrap.mean()

            delta_mean_percentage = 100*(observed_proportions-expected_proportions_bootstrap)/observed_proportions


            # Append to bootstrap results
            bootstrap_results.append(delta_mean_percentage)
            expected_proportions_bootstrap_list.append(expected_proportions_bootstrap)

            # Append to bootstrap results
            observed.append(observed_proportions)
            expected.append(expected_proportions_bootstrap)
            
        bootstrap_df[f'{year}_F/F_bootstrap_vals'] = bootstrap_results
        bootstrap_df[f'{year}_F/F_observed'] = observed
        bootstrap_df[f'{year}_F/F_expected'] = expected_proportions_bootstrap_list
    
        # Calculate the 95% confidence interval
        se_means = np.std(bootstrap_results)
        ff_confidence_interval = (np.mean(bootstrap_results)-1.96*se_means/(n_bootstrap ** 0.5),np.mean(bootstrap_results)+1.96*se_means/(n_bootstrap ** 0.5))
        ff_conf_ints.append((ff_confidence_interval[1]-ff_confidence_interval[0])/2) 
        # Print the confidence interval
        print(f'95% Confidence Interval: {ff_confidence_interval}')
        ff_p_val = calculate_p_value(bootstrap_results)
        ff_p_vals.append(ff_p_val)
        print(f'p-value: {ff_p_val}')
    
    
      
        
        
        
        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']!='M/M')]
        n_bootstraped = 1000
        observed_values = test_data['M/M']/test_data['Author Count']
        # Bootstrap resampling
        bootstrap_samples = np.random.choice(observed_values, size=(len(observed_values), n_bootstrap), replace=True)
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        # Calculate the 95% confidence interval
        mm_obs_ci.append(np.percentile(bootstrap_means, [2.5, 97.5]))
        mm_obs_p_val = calculate_p_value(bootstrap_samples)
        mm_obs_p_values.append(mm_obs_p_val)

        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']!='M/M')]
        n_bootstraped = 1000
        observed_values = test_data['M/F']/test_data['Author Count']
        # Bootstrap resampling
        bootstrap_samples = np.random.choice(observed_values, size=(len(observed_values), n_bootstrap), replace=True)
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        # Calculate the 95% confidence interval
        mf_obs_ci.append(np.percentile(bootstrap_means, [2.5, 97.5]))
        mf_obs_p_val = calculate_p_value(bootstrap_samples)
        mf_obs_p_values.append(mf_obs_p_val)



        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']!='M/M')]
        n_bootstraped = 1000
        observed_values = test_data['F/M']/test_data['Author Count']
        # Bootstrap resampling
        bootstrap_samples = np.random.choice(observed_values, size=(len(observed_values), n_bootstrap), replace=True)
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        # Calculate the 95% confidence interval
        fm_obs_ci.append(np.percentile(bootstrap_means, [2.5, 97.5]))
        fm_obs_p_val = calculate_p_value(bootstrap_samples)
        fm_obs_p_values.append(fm_obs_p_val)



        test_data = model_data[(model_data['Publication Year'] == year) & (model_data['Gender']!='M/M')]
        n_bootstraped = 1000
        observed_values = test_data['F/F']/test_data['Author Count']
        # Bootstrap resampling
        bootstrap_samples = np.random.choice(observed_values, size=(len(observed_values), n_bootstraped), replace=True)
        # Calculate the mean for each bootstrap sample
        bootstrap_means = np.mean(bootstrap_samples, axis=0)
        # Calculate the 95% confidence interval
        ff_obs_ci.append(np.percentile(bootstrap_means, [2.5, 97.5]))
        ff_obs_p_val = calculate_p_value(bootstrap_samples)
        ff_obs_p_values.append(ff_obs_p_val)
    
        bootstrap_df.to_csv(f'/Users/akashrathi/Desktop/BrownMed/{bootstrap_df_name}', index=False)

fig4_gam_model(model_data, bootstrap_df, 'fig4_bootstrap_df2', 2007, 2023, n_bootstrap=1000)

MM_bootstrap_vals = []
for col in bootstrap_df.columns:
  if 'M/M_bootstrap' in col:
    MM_bootstrap_vals.append(bootstrap_df[col].mean())
MM_bootstrap_vals = [x for x in MM_bootstrap_vals if str(x) != 'nan']
#MM_errors = np.std(MM_bootstrap_vals, axis=0) / np.sqrt(len(MM_bootstrap_vals))


MF_bootstrap_vals = []
for col in bootstrap_df.columns:
  if 'M/F_bootstrap' in col:
    MF_bootstrap_vals.append(bootstrap_df[col].mean())
MF_bootstrap_vals = [x for x in MF_bootstrap_vals if str(x) != 'nan']
#MF_errors = np.std(MF_bootstrap_vals, axis=0) / np.sqrt(len(MF_bootstrap_vals))

FM_bootstrap_vals = []
for col in bootstrap_df.columns:
  if 'F/M_bootstrap' in col:
    FM_bootstrap_vals.append(bootstrap_df[col].mean())
FM_bootstrap_vals = [x for x in FM_bootstrap_vals if str(x) != 'nan']
#FM_errors = np.std(FM_bootstrap_vals, axis=0) / np.sqrt(len(FM_bootstrap_vals))

FF_bootstrap_vals = []
for col in bootstrap_df.columns:
  if 'F/F_bootstrap' in col:
    FF_bootstrap_vals.append(bootstrap_df[col].mean())
FF_bootstrap_vals = [x for x in FF_bootstrap_vals if str(x) != 'nan']
#FF_errors = np.std(FF_bootstrap_vals, axis=0) / np.sqrt(len(FF_bootstrap_vals))





MM_expected_vals = []
for col in bootstrap_df.columns:
  if 'M/M_expected' in col:
    MM_expected_vals.append(bootstrap_df[col].mean())
MM_expected_vals = [x for x in MM_expected_vals if str(x) != 'nan']
#MM_errors = np.std(MM_bootstrap_vals, axis=0) / np.sqrt(len(MM_bootstrap_vals))


MF_expected_vals = []
for col in bootstrap_df.columns:
  if 'M/F_expected' in col:
    MF_expected_vals.append(bootstrap_df[col].mean())
MF_expected_vals = [x for x in MF_expected_vals if str(x) != 'nan']
#MF_errors = np.std(MF_bootstrap_vals, axis=0) / np.sqrt(len(MF_bootstrap_vals))

FM_expected_vals = []
for col in bootstrap_df.columns:
  if 'F/M_expected' in col:
    FM_expected_vals.append(bootstrap_df[col].mean())
FM_expected_vals = [x for x in FM_expected_vals if str(x) != 'nan']
#FM_errors = np.std(FM_bootstrap_vals, axis=0) / np.sqrt(len(FM_bootstrap_vals))

FF_expected_vals = []
for col in bootstrap_df.columns:
  if 'F/F_expected' in col:
    FF_expected_vals.append(bootstrap_df[col].mean())
FF_expected_vals = [x for x in FF_expected_vals if str(x) != 'nan']
#FF_errors = np.std(FF_bootstrap_vals, axis=0) / np.sqrt(len(FF_bootstrap_vals))




MM_observed_vals = []
for col in bootstrap_df.columns:
  if 'M/M_observed' in col:
    MM_observed_vals.append(bootstrap_df[col].mean())
MM_observed_vals = [x for x in MM_observed_vals if str(x) != 'nan']
#MM_errors = np.std(MM_bootstrap_vals, axis=0) / np.sqrt(len(MM_bootstrap_vals))


MF_observed_vals = []
for col in bootstrap_df.columns:
  if 'M/F_observed' in col:
    MF_observed_vals.append(bootstrap_df[col].mean())
MF_observed_vals = [x for x in MF_observed_vals if str(x) != 'nan']
#MF_errors = np.std(MF_bootstrap_vals, axis=0) / np.sqrt(len(MF_bootstrap_vals))

FM_observed_vals = []
for col in bootstrap_df.columns:
  if 'F/M_observed' in col:
    FM_observed_vals.append(bootstrap_df[col].mean())
FM_observed_vals = [x for x in FM_observed_vals if str(x) != 'nan']
#FM_errors = np.std(FM_bootstrap_vals, axis=0) / np.sqrt(len(FM_bootstrap_vals))

FF_observed_vals = []
for col in bootstrap_df.columns:
  if 'F/F_observed' in col:
    FF_observed_vals.append(bootstrap_df[col].mean())
FF_observed_vals = [x for x in FF_observed_vals if str(x) != 'nan']

# MM observed vs. expected
mm_obs_ci_error = []
for i in mm_obs_ci:
    mm_obs_ci_error.append((np.diff(i)/2)[0])

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

rounded_years = np.ceil(years)
rounded_years = rounded_years.astype(int)

# Set the rounded values as tick labels
plt.xticks(years, rounded_years)
plt.xticks(np.arange(rounded_years.min(), rounded_years.max() + 1, 2))

plt.plot(years, MM_observed_vals, label='M/M', marker='o', color = 'navy')
plt.plot(years, MM_expected_vals, label='M/M', marker='o', color = 'navy', linestyle='--')

plt.errorbar(years, MM_observed_vals, yerr=mm_obs_ci_error, label='M/M', marker='o', linestyle='', capsize=5, color = 'navy')
#plt.errorbar(years, MM_expected_vals, yerr=mm_errors, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')


# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.ylim()
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: female or female (MM)')
plt.ylim(0.60,0.95)

plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_ForF_MM', format='pdf')

# Show the plot
plt.show()


# MF observed vs. expected
mf_obs_ci_error = []
for i in mf_obs_ci:
    mf_obs_ci_error.append((np.diff(i)/2)[0])

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

rounded_years = np.ceil(years)
rounded_years = rounded_years.astype(int)

# Set the rounded values as tick labels
plt.xticks(years, rounded_years)
plt.xticks(np.arange(rounded_years.min(), rounded_years.max() + 1, 2))

plt.plot(years, MF_observed_vals, label='M/M', marker='o', color = 'turquoise')
plt.plot(years, MF_expected_vals, label='M/M', marker='o', color = 'turquoise', linestyle='--')

plt.errorbar(years, MF_observed_vals, yerr=mf_obs_ci_error, label='M/M', marker='o', linestyle='', capsize=5, color = 'turquoise')
#plt.errorbar(years, MF_exp_line, yerr=MF_errors, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')


# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.ylim(-1,1)
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: female or female')
plt.ylim(0,0.35)

plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_ForF_MF', format='pdf')

# Show the plot
plt.show()

print('observed slope CI:',(np.polyfit(years, MF_observed_vals, 1)[0]-np.mean(mf_obs_ci_error),
                           np.polyfit(years, MF_observed_vals, 1)[0]+np.mean(mf_obs_ci_error)))
print('observed slope p-value:',np.mean(mf_obs_p_values))
print('\n')
print('expected slope:',(np.polyfit(years, MF_expected_vals, 1)[0]))

# FM observed vs. expected


fm_obs_ci_error = []
for i in fm_obs_ci:
    fm_obs_ci_error.append((np.diff(i)/2)[0])

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

rounded_years = np.ceil(years)
rounded_years = rounded_years.astype(int)

# Set the rounded values as tick labels
plt.xticks(years, rounded_years)
plt.xticks(np.arange(rounded_years.min(), rounded_years.max() + 1, 2))

plt.plot(years, FM_observed_vals, label='M/M', marker='o', color = 'orange')
plt.plot(years, FM_expected_vals, label='M/M', marker='o', color = 'orange', linestyle='--')

plt.errorbar(years, FM_observed_vals, yerr=fm_obs_ci_error, label='M/M', marker='o', linestyle='', capsize=5, color = 'orange')
#plt.errorbar(years, MF_exp_line, yerr=MF_errors, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')


# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.ylim(-1,1)
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: female or female')
plt.ylim(0,0.35)


plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_ForF_FM', format='pdf')

# Show the plot
plt.show()

print('observed slope CI:',(np.polyfit(years, FM_observed_vals, 1)[0]-np.mean(fm_obs_ci_error),
                           np.polyfit(years, FM_observed_vals, 1)[0]+np.mean(fm_obs_ci_error)))
print('observed slope p-value:',np.mean(fm_obs_p_values))
print('\n')
print('expected slope:',(np.polyfit(years, FM_expected_vals, 1)[0]))

# FF observed vs. expected

ff_obs_ci_error = []
for i in ff_obs_ci:
    ff_obs_ci_error.append((np.diff(i)/2)[0])

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

plt.plot(years, FF_observed_vals, label='M/M', marker='o', color = 'green')
plt.plot(years, FF_expected_vals, label='M/M', marker='o', color = 'green', linestyle='--')

plt.errorbar(years, FF_observed_vals, yerr=ff_obs_ci_error, label='M/M', marker='o', linestyle='', capsize=5, color = 'green')
#plt.errorbar(years, MF_exp_line, yerr=MF_errors, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')


# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.ylim(-0.07,0.30)
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: female or female')
plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_ForF_FF', format='pdf')

# Show the plot
plt.show()

print('observed slope CI:',(np.polyfit(years, FF_observed_vals, 1)[0]-np.mean(ff_obs_ci_error),
                           np.polyfit(years, FF_observed_vals, 1)[0]+np.mean(ff_obs_ci_error)))
print('observed slope p-value:',np.mean(ff_obs_p_values))
print('\n')
print('expected slope:',(np.polyfit(years, FF_expected_vals, 1)[0]))


# All gender combinations

# Years from 2006 to 2023
years = list(range(2007, 2024))

# Plot each list on the same line plot
plt.figure(figsize=(10, 6))

rounded_years = np.ceil(years)
rounded_years = rounded_years.astype(int)

# Set the rounded values as tick labels
plt.xticks(years, rounded_years)
plt.xticks(np.arange(rounded_years.min(), rounded_years.max() + 1, 2))

plt.plot(years, MM_bootstrap_vals, label='M/M', marker='o', color = 'navy')
plt.plot(years, MF_bootstrap_vals, label='M/F', marker='o', color = 'turquoise')
plt.plot(years, FM_bootstrap_vals, label='F/M', marker='o', color = 'orange')
plt.plot(years, FF_bootstrap_vals, label='F/F', marker='o', color = 'green')

plt.errorbar(years, MM_bootstrap_vals, yerr=mm_conf_ints, label='M/M', marker='o', linestyle='', capsize=5, color = 'navy')
plt.errorbar(years, MF_bootstrap_vals, yerr=mf_conf_ints, label='M/F', marker='o', linestyle='', capsize=5, color = 'turquoise')
plt.errorbar(years, FM_bootstrap_vals, yerr=fm_conf_ints, label='F/M', marker='o', linestyle='', capsize=5, color = 'orange')
plt.errorbar(years, FF_bootstrap_vals, yerr=ff_conf_ints, label='F/F', marker='o', linestyle='', capsize=5, color = 'green')

# Add a dotted horizontal line at y=0
plt.axhline(y=0, linestyle='--', color='black', linewidth=1)

# Set plot labels and title
plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
plt.xlabel('Year')
plt.ylabel('Over- and undercitation (%)')
plt.title('Citing authors: female or female')
#plt.ylim(-9,1)

plt.savefig('/Users/akashrathi/Desktop/BrownMed/fig4_overall_ex_ForF', format='pdf')

# Show the plot
plt.show()

print('MM slope CI:',(np.polyfit(years, MM_bootstrap_vals, 1)[0]-np.mean(mm_conf_ints),
                           np.polyfit(years, MM_bootstrap_vals, 1)[0]+np.mean(mm_conf_ints)))
print('MM slope p-value:',np.mean(mm_p_vals))

print('MF slope CI:',(np.polyfit(years, MF_bootstrap_vals, 1)[0]-np.mean(mf_conf_ints),
                           np.polyfit(years, MF_bootstrap_vals, 1)[0]+np.mean(mf_conf_ints)))
print('MF slope p-value:',np.mean(mf_p_vals))
print('FM slope CI:',(np.polyfit(years, FM_bootstrap_vals, 1)[0]-np.mean(fm_conf_ints),
                           np.polyfit(years, FM_bootstrap_vals, 1)[0]+np.mean(fm_conf_ints)))
print('FM slope p-value:',np.mean(fm_p_vals))
print('FF slope CI:',(np.polyfit(years, FF_bootstrap_vals, 1)[0]-np.mean(ff_conf_ints),
                           np.polyfit(years, FF_bootstrap_vals, 1)[0]+np.mean(ff_conf_ints)))
print('FF slope p-value:',np.mean(ff_p_vals))