#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 17:34:16 2026

@author: tim
"""

# Mixed linear Model for Ephys Env vs PT manuscript:

# code from TWF paper was modified for that:
# %%

'''
Adjust the data accordingly -> 
'''
import config  # data/figure folders, see config.py
import re

import statsmodels.api as sm
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
# from statsmodels.stats.api import anova_lm
import sys
# import matplotlib
# import glob
from scipy import stats 
# import researchpy as rp
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


import matplotlib.patches as patches
figurepath = config.FIGURE_ROOT+'/'
figurepath = config.FIGURE_ROOT+'/'

analysisRoot = config.ANALYSIS_ROOT
dataRoot = config.DATA_ROOT
dataFiles = pd.read_csv(analysisRoot+'/datafileInventory.csv')
# dataFiles = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/datafileInventory_Salt_Beaker_Agar.csv')
# baselinerecordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar_baseline.csv')
# recordingSites = pd.read_csv(analysisRoot+'/Salt_Beaker_CSV_Files/recordingsiteInventory_Salt_Beaker_Agar.csv')
baselinerecordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory_baseline.csv')
recordingSites = pd.read_csv(config.ANALYSIS_ROOT+'/recordingsiteInventory.csv')

# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')

# str_recordingsites = analysisRoot+'/recordingsiteInventory_EVE.csv'
str_recordingsites = analysisRoot+'/recordingsiteInventory.csv'


# dataFiles = pd.read_csv(analysisRoot+'/datafileInventory_V3.csv')
# dataFiles = pd.read_csv(analysisRoot+'/drecordingsiteInventory_Salt_Beaker_VE.csv')


# str_recordingsites = analysisRoot+'/recordingsiteInventory_EVE.csv'
# str_recordingsites = analysisRoot+'/recordingsiteInventoryV3.csv'
# str_recordingsites = analysisRoot+'/datafileInventory_Salt_Beaker_V2.csv'

recordingSites = pd.read_csv(str_recordingsites)



# fig_6_bar, axes_6_bar = plt.subplots (2, 3, figsize = (7.2, 6), sharey = True)


# Function to check if 'EVE' is in the string using regular expressions
def contains_eve(input_string):
    # Regular expression to search for 'EVE' (case-sensitive)
    pattern = r'_EVE'
    
    # Use re.search to check if 'EVE' is found in the input string
    if re.search(pattern, input_string):
        return True
    else:
        return False

corhorts = dataFiles.condition.unique()
corhorts = corhorts[:3]

animDir = dataFiles.animDir.unique()
# Get the headers
headers = recordingSites.columns
Data_header = headers[4:]       ################Careful here- old code was 6: !!!
significance_headers = [col for col in Data_header if "significance" in col]
VE_headers = [
    col for col in Data_header if "significance" not in col and "pValue" not in col]
pValue_headers = [col for col in Data_header if "pValue" in col]

# jusit a quick and dirty way to get the headers for a plot...
Headers_Plot = []
# This is to get the VE letters --> Jan wated that for the plot
VE_kind = [header.split('_', 2)[-1] for header in VE_headers]
VE_kind = [f"VE {entry[2:].upper()}" for entry in VE_kind]
Dur_kind = [header.split('_', 1)[0] for header in VE_headers]
pps_kind = [header.split('_')[1] for header in VE_headers]


for i in (range(len(VE_headers))):
    # print (head)

    Headers_Plot.append(VE_kind[i] + ' ' +
                        Dur_kind[i][3:] + ' s, ' + pps_kind[i][4:-2] + ' pps')

Anims = []
for anim in animDir:
    # print(anim)
    filenames = dataFiles[(dataFiles['animDir'] == anim)& (dataFiles['Cleaned'] == 1)]['fileName']
    if len(filenames) == 0:
        continue
    sub_recordingSites = []
    for filename in filenames:
        sub_recordingSites.append(
            recordingSites[recordingSites['fileName'] == filename])
    sub_recordingSites = pd.concat(sub_recordingSites, ignore_index=True)
    # sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]
    Anims.append(sub_recordingSites)

Anims_baseline = []
for anim in animDir:
    filenames = dataFiles[(dataFiles['animDir'] == anim)& (dataFiles['Cleaned'] == 1)]['fileName']
    if len(filenames) == 0:
        continue
    sub_recordingSites_baseline = []
    for filename in filenames:
        sub_recordingSites_baseline.append(
            baselinerecordingSites[baselinerecordingSites['fileName'] == filename])
    sub_recordingSites_baseline = pd.concat(sub_recordingSites_baseline, ignore_index=True)
    # sub_recordingSites = sub_recordingSites[sub_recordingSites['Responsive'] == 1]
    Anims_baseline.append(sub_recordingSites_baseline)

# def create_subplot ():
#     fig, axes = plt.subplots(3, 1)


ALL_DATA = []
SIG_DATA = []

All_VE_Data_Outside = []
count_outside = 0

for corhNum, corhort in enumerate(corhorts):
    # if type(Onlycohort) == int :
    # if corhNum != Onlycohort and OnlycohortStr != 'n' and OnlycohortStr != 'all':
        # plt.figure()
    # print(corhort)
    # all_units = []
    All_data = [] #holds all animals penetrations with all VE headers
    Sig_data = [] #holds the significance vals 0 or 1
    Baseline_Sig_data = []
    No_Animals = 0
    # All_labels = []
    for idx, anim in enumerate(Anims):
        # print(idx)
        # if idx == 7:# or idx == 19 or idx == 13:
        #     continue
        # print (Anims[idx].condition.iloc[0])
        if Anims[idx].cohort.iloc[0] == corhort:
            No_Animals += 1
                # print(corhort)
                # Check here how ,many UNITS!
    
                # Units = len(anim)
                # all_units.append(Units)
            All_data.append(anim[VE_headers]*100)
            Sig_data.append(anim[significance_headers])
    
    for idx_base, anim_base in enumerate(Anims_baseline):
        # if idx_base == 7:# or idx == 19 or idx == 13:
        #     continue
        if Anims_baseline[idx_base].cohort.iloc[0] == corhort:
            Baseline_Sig_data.append(anim_base[significance_headers])


            # All_labels.append(Headers_Plot)
            
            
    ######################################################
    ######## Remove if baseline VE is significant ########
    ######################################################
    # --> Changes the values with ones to a zero 
    
    Baseline_list = []
    for i, pandas in enumerate (Baseline_Sig_data):
        nested_indices = []
        for i in range(0, len(pandas.columns), 2):
            col_pair = pandas.columns[i:i+2]
            mask = (pandas[col_pair[0]] == 1) | (pandas[col_pair[1]] == 1)
            indices = pandas.index[mask].tolist()
            nested_indices.append(indices)
        Baseline_list.append(nested_indices)
    
    for i, pandas in enumerate (Baseline_Sig_data):
        # print(i)
        for column, index_remove in enumerate (Baseline_list[i]):
            # print(column) #--> 0-5
            # print(index_remove) #--> liste removed indices
            LoopJump = column * 2
            cols = Baseline_Sig_data[i].columns[LoopJump:LoopJump+2]
            # print (cols) #--> header list - each header and the following.
            # Now replace these values with nan and do the same for all data --> solved :) 
            Sig_data[i].loc[index_remove, cols] = np.nan
            # All_data[i].loc[index_remove, cols] = np.nan
        
    for i, pandas in enumerate (Baseline_Sig_data):
        # print(i)
        for column, index_remove in enumerate (Baseline_list[i]):
            # print(column) #--> 0-5
            # print(index_remove) #--> liste removed indices
            LoopJump = column * 2
            cols = All_data[i].columns[LoopJump:LoopJump+2]
            # print (cols) #--> header list - each header and the following.
            # Now replace these values with nan and do the same for all data --> solved :) 
            All_data[i].loc[index_remove, cols] = np.nan

    #New counting function also:
    all_units = []
    for i, pandas in enumerate (Sig_data):
        rows_with_entries = pandas.notna().any(axis=1).sum()
        all_units.append(rows_with_entries)
    No_All_Units = np.sum(all_units)    
    print('Number of all units for this cohort', No_All_Units)

    ############################################################################################
    ############################################################################################
    ############################################################################################
    
    Combined_Data = pd.concat(All_data)
    Combined_Data = [Combined_Data[col].dropna() for col in Combined_Data.columns]
    All_VE_Data_Outside.append(Combined_Data)

    ALL_DATA.append(All_data)
    SIG_DATA.append(Sig_data)


    ############################################################################################
    ############################################################################################
    ############################################################################################
    
    Combined_Data_Sig = pd.concat(Sig_data)
    Combined_Data_Sig = [Combined_Data_Sig[col].dropna() for col in Combined_Data_Sig.columns]
    if corhort == 'ND':
        print(Combined_Data)
            
               
    Percentage_vals = []
    Units_cond = []
    for i_Sig, val_Sig in enumerate(Combined_Data_Sig):
            # print(i_Sig)
        for i, val in enumerate(Combined_Data):
                # print(val)
            if i == i_Sig:
                n = len(val_Sig)
                n2 = len(val)
                Sig = sum(val_Sig)
                
                # for index, value in val_Sig.items():  # .iteritems() also works
                #     if index == 0:
                #         count += 1
                        
                Percentage = round((Sig/n)*100, 2)
                Percentage_vals.append(Percentage)
                Units_cond.append(n)
                    #Plot percentage units here if neede:
                    # plt.text(
                    #     i_Sig + 1, -1, f"Units : {n}", ha='center', fontsize=10, color='blue')
                    # # plt.text (i_Sig + 1, -2, f"Units2 : {n2}", ha='center', fontsize=10, color='blue')
                    # plt.text(
                    #     i_Sig + 1, -3, f"% : {Percentage}", ha='center', fontsize=10, color='blue')
            
    # for i_6_bar, ax_6_bar in enumerate (axes_6_bar.flatten()):
    #         # print(i_6_bar)
    #         # print('Loopy', i_6_bar % len(corhorts))
    #     headers_6_bar = []
    #     percentage_6_bar = []
    #     count = 0
    #     for i_perc, perc in enumerate(Percentage_vals):
    #             # print(i_perc)
    #         headers_6_bar.append(Headers_Plot[count: count+2])
    #         percentage_6_bar.append(Percentage_vals[count: count+2])
    #         count += 2
    #         if count >= 12:
    #             break
            
    #     modified_headers_6_bar = [headers_6_bar[0], headers_6_bar[2], headers_6_bar[4], headers_6_bar[1], headers_6_bar[3], headers_6_bar[5]]
    #     modified_percentage_6_bar = [percentage_6_bar[0], percentage_6_bar[2], percentage_6_bar[4], percentage_6_bar[1], percentage_6_bar[3], percentage_6_bar[5]]
    #     Adjusted_header_plot = ['10 ms, 900 pps', '50 ms, 900 pps', '200 ms, 900 pps', '10 ms, 4500 pps', '50 ms, 4500 pps', '200 ms, 4500 pps']
            
# %%


    
#Extract the data for every single animal:



Cohort_list = []

for cohort in range(len(ALL_DATA)):
    Anim_list = []  
    for anim_idx, data in enumerate (ALL_DATA[cohort]):

        # if anim_idx == 4:
        # print(Anims[anim_idx][0].fileName)
        data = [data[col].dropna() for col in data.columns]
        Anim = []  
        for i, dat in enumerate(data):
                # print(len(dat))
                # print(i)
            # print(len(data))
            Anim.append(data[i].name)
            Anim.append(np.median(dat))
        Anim_list.append(Anim)
    Cohort_list.append(Anim_list)

#Make csv file for GEE model:
import csv
import re

# Animal names for each cohort
animal_names = [
    ['2021_12_16', '2022_01_11', '2022_02_18', '2022_02_21', '2022_08_22', '2022_03_02'],
    ['2022_01_24', '2022_01_25', '2022_01_27', '2022_08_24', '2023_02_16', '2023_02_21', '2023_02_23', '2023_03_03'],
    ['CI2225_ITD_ENV_vs_PT_CI_1', 'CI2226_ITD_ENV_vs_PT_CI_1', 'CI2302_ITD_ENV_vs_PT_CI_1', 
     'CI2303_ITD_ENV_vs_PT_CI_1', 'CI2304_ITD-_ENV_vs_PT_CI_1', 'CI2229_ITD-_ENV_vs_PT_CI_1']
]

def process_cohort_data(Cohort_list):
    """Process Cohort_list with animal names."""
    cohort_names = ['NH', 'ND', 'ITD+']
    all_rows = []
    row_index = 0
    
    # Helper function to process data for a specific stimulus type
    def process_stimulus_type(stim_type, stim_code):
        nonlocal row_index
        rows = []
        
        for cohort_idx, cohort_data in enumerate(Cohort_list):
            cohort_name = cohort_names[cohort_idx] if cohort_idx < len(cohort_names) else f'Cohort_{cohort_idx}'
            print(cohort_name)
            
            for animal_idx, animal_data in enumerate(cohort_data):
                # Get animal name
                if cohort_idx < len(animal_names) and animal_idx < len(animal_names[cohort_idx]):
                    animal_name = animal_names[cohort_idx][animal_idx]
                    print(animal_name)
                else:
                    animal_name = ''
                
                # Collect values for this stimulus type
                values = {}
                for i in range(0, len(animal_data), 2):
                    header = animal_data[i]
                    value = animal_data[i + 1]
                    
                    if stim_code in header:
                        dur_match = re.search(r'Dur(\d+\.\d+)', header)
                        rate_match = re.search(r'Rate(\d+\.\d+)', header)
                        
                        if dur_match and rate_match:
                            duration = float(dur_match.group(1))
                            rate = float(rate_match.group(1))
                            values[(rate, duration)] = value
                
                # Sort and add rows
                for rate, duration in sorted(values.keys(), key=lambda x: (-x[1], x[0])):
                    rows.append({
                        'Pulserate': rate,
                        'stimDuration': duration,
                        'Animal': animal_name,
                        'Cohort': cohort_name,
                        'VE': values[(rate, duration)],
                        'stimulusType': stim_type,
                        'index': row_index
                    })
                    row_index += 1
        
        return rows
    
    # Process PT first
    all_rows.extend(process_stimulus_type('PT', '_VEpt'))
    
    # Process ENV second
    all_rows.extend(process_stimulus_type('ENV', '_VEenv'))
    
    return all_rows

def save_to_csv(data, filename='cohort_analysis.csv'):
    """Save processed data to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Pulserate', 'stimDuration', 'Animal', 'Cohort', 'VE', 'stimulusType', 'index'])
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data saved to {filename}")
    return filename

# To use this code:
# 1. Make sure Cohort_list is defined
# 2. Run:
processed_data = process_cohort_data(Cohort_list)
save_to_csv(processed_data, 'your_output_file.csv')

#Extract the data for the cohorts- median:
# PT_all = []   
# ENV_all = []
         
# for i in range (len(All_VE_Data_Outside)):
#     for ii in range (0, 12, 2):
#         col1 = All_VE_Data_Outside[i][ii]
#         col2 = All_VE_Data_Outside[i][ii+1]
        
#         PT_all.append(np.median(col1))
#         ENV_all.append(np.median(col2))


# %%
# Ve values vs. log pulserate
#Model code:
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import matplotlib.pyplot as plt
# from patsy import dmatrix
# from scipy import stats

# # Load your data
# df = pd.read_csv(config.ANALYSIS_ROOT+'/MEM_ENVvsPT.csv')


# # Clean column names (remove extra spaces)
# df.columns = df.columns.str.strip()

# print("Data shape:", df.shape)
# print("\nColumns:", df.columns.tolist())

# # Clean up data
# df_clean = df.copy()

# # Rename column with typo
# df_clean = df_clean.rename(columns={'Aimal': 'Animal'})

# # Convert data types
# df_clean['Animal'] = df_clean['Animal'].astype(str)
# df_clean['Cohort'] = df_clean['Cohort'].astype('category')

# # Extract numeric value from Pulserate (remove ' pps')
# df_clean['Pulserate_numeric'] = df_clean['Pulserate'].str.replace(' pps', '').astype(float)

# # Create log_PulseRate
# df_clean['log_PulseRate'] = np.log(df_clean['Pulserate_numeric'])

# print("\n" + "="*80)
# print("DATA PREPARATION")
# print("="*80)
# print(f"\nPulserate values: {sorted(df_clean['Pulserate_numeric'].unique())}")
# print(f"log_PulseRate range: [{df_clean['log_PulseRate'].min():.2f}, {df_clean['log_PulseRate'].max():.2f}]")
# print(f"Cohorts: {sorted(df_clean['Cohort'].unique())}")
# print(f"Number of animals: {df_clean['Animal'].nunique()}")

# # ============================================================================
# # ANALYSIS 1: VE PT - Modeling how PT depends on log_PulseRate
# # ============================================================================
# print("\n" + "="*80)
# print("ANALYSIS 1: VE PT MODEL")
# print("="*80)

# # Remove rows with missing VE PT values if any
# df_pt = df_clean.dropna(subset=['VE PT']).copy()

# # Fit the model for VE PT
# model_pt = smf.mixedlm(
#     formula="Q('VE PT') ~ log_PulseRate * C(Cohort)",
#     data=df_pt,
#     groups=df_pt["Animal"],
#     re_formula="log_PulseRate"  # Random slopes per Animal
# )

# print("\nFitting VE PT model...")
# fittedModel_pt = model_pt.fit(method='lbfgs', maxiter=1000)

# print("\n" + "="*80)
# print("VE PT MODEL RESULTS")
# print("="*80)
# print(fittedModel_pt.summary())

# # Check interaction terms for VE PT
# interaction_terms_pt = [p for p in fittedModel_pt.params.index if ':' in p]
# print("\n1. INTERACTION TERMS FOR VE PT (tests if slopes differ by Cohort):")
# for term in interaction_terms_pt:
#     if term in fittedModel_pt.pvalues:
#         pval = fittedModel_pt.pvalues[term]
#         estimate = fittedModel_pt.params[term]
#         se = fittedModel_pt.bse[term]
        
#         # Extract cohort names
#         parts = term.split(':')
#         pulse_part = [p for p in parts if 'log_PulseRate' in p][0]
#         cohort_part = [p for p in parts if 'Cohort' in p][0]
        
#         # Clean up cohort name
#         cohort_name = cohort_part.split('[')[-1].rstrip(']').replace('T.', '')
        
#         print(f"\n   {term}")
#         print(f"   β = {estimate:.4f} ± {se:.4f}")
#         print(f"   p = {pval:.4f}", end="")
        
#         if pval < 0.001:
#             print("  ***")
#         elif pval < 0.01:
#             print("  **")
#         elif pval < 0.05:
#             print("  *")
#         else:
#             print("")
        
#         print(f"   Interpretation: The slope for Cohort '{cohort_name}' differs from")
#         print(f"   the reference Cohort by {estimate:.4f} units per log(PulseRate)")

# # ============================================================================
# # ANALYSIS 2: VE ENV - Modeling how ENV depends on log_PulseRate
# # ============================================================================
# print("\n" + "="*80)
# print("ANALYSIS 2: VE ENV MODEL")
# print("="*80)

# # Remove rows with missing VE ENV values if any
# df_env = df_clean.dropna(subset=['VE ENV']).copy()

# # Fit the model for VE ENV
# model_env = smf.mixedlm(
#     formula="Q('VE ENV') ~ log_PulseRate * C(Cohort)",
#     data=df_env,
#     groups=df_env["Animal"],
#     re_formula="log_PulseRate"  # Random slopes per Animal
# )

# print("\nFitting VE ENV model...")
# fittedModel_env = model_env.fit(method='lbfgs', maxiter=1000)

# print("\n" + "="*80)
# print("VE ENV MODEL RESULTS")
# print("="*80)
# print(fittedModel_env.summary())

# # Check interaction terms for VE ENV
# interaction_terms_env = [p for p in fittedModel_env.params.index if ':' in p]
# print("\n2. INTERACTION TERMS FOR VE ENV (tests if slopes differ by Cohort):")
# for term in interaction_terms_env:
#     if term in fittedModel_env.pvalues:
#         pval = fittedModel_env.pvalues[term]
#         estimate = fittedModel_env.params[term]
#         se = fittedModel_env.bse[term]
        
#         # Extract cohort names
#         parts = term.split(':')
#         pulse_part = [p for p in parts if 'log_PulseRate' in p][0]
#         cohort_part = [p for p in parts if 'Cohort' in p][0]
        
#         # Clean up cohort name
#         cohort_name = cohort_part.split('[')[-1].rstrip(']').replace('T.', '')
        
#         print(f"\n   {term}")
#         print(f"   β = {estimate:.4f} ± {se:.4f}")
#         print(f"   p = {pval:.4f}", end="")
        
#         if pval < 0.001:
#             print("  ***")
#         elif pval < 0.01:
#             print("  **")
#         elif pval < 0.05:
#             print("  *")
#         else:
#             print("")
        
#         print(f"   Interpretation: The slope for Cohort '{cohort_name}' differs from")
#         print(f"   the reference Cohort by {estimate:.4f} units per log(PulseRate)")

# # ============================================================================
# # VISUALIZATION
# # ============================================================================
# print("\n" + "="*80)
# print("VISUALIZATION")
# print("="*80)

# # Create figure with subplots
# fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# fig.suptitle('VE Values vs log(PulseRate) by Cohort', fontsize=16, fontweight='bold')

# # Color scheme for cohorts
# cohorts = sorted(df_clean['Cohort'].unique())
# cohort_colors = plt.cm.Set1(np.linspace(0, 1, len(cohorts)))
# cohort_color_dict = dict(zip(cohorts, cohort_colors))

# # Plot 1: VE PT raw data with regression lines
# for idx, cohort in enumerate(cohorts):
#     cohort_data = df_pt[df_pt['Cohort'] == cohort]
    
#     # Plot raw data
#     axes[0, 0].scatter(cohort_data['log_PulseRate'], cohort_data['VE PT'],
#                        alpha=0.5, color=cohort_colors[idx], 
#                        label=f'{cohort}', s=40)
    
#     # Create prediction grid for this cohort
#     x_min, x_max = cohort_data['log_PulseRate'].min(), cohort_data['log_PulseRate'].max()
#     x_grid = np.linspace(x_min, x_max, 50)
    
#     # Create prediction data
#     pred_data = pd.DataFrame({
#         'log_PulseRate': x_grid,
#         'Cohort': [cohort] * 50,
#         'Animal': [cohort_data['Animal'].iloc[0]] * 50  # Use first animal for prediction
#     })
    
#     # Get predictions (population average)
#     pred_data['predicted'] = fittedModel_pt.predict(pred_data)
    
#     # Plot regression line
#     axes[0, 0].plot(x_grid, pred_data['predicted'], 
#                     color=cohort_colors[idx], linewidth=2.5, 
#                     label=f'{cohort} fit')

# axes[0, 0].set_xlabel('log(PulseRate)')
# axes[0, 0].set_ylabel('VE PT')
# axes[0, 0].set_title('VE PT vs log(PulseRate) by Cohort')
# axes[0, 0].legend(loc='best', fontsize='small')
# axes[0, 0].grid(True, alpha=0.3)

# # Plot 2: VE ENV raw data with regression lines
# for idx, cohort in enumerate(cohorts):
#     cohort_data = df_env[df_env['Cohort'] == cohort]
    
#     # Plot raw data
#     axes[0, 1].scatter(cohort_data['log_PulseRate'], cohort_data['VE ENV'],
#                        alpha=0.5, color=cohort_colors[idx], 
#                        label=f'{cohort}', s=40)
    
#     # Create prediction grid for this cohort
#     x_min, x_max = cohort_data['log_PulseRate'].min(), cohort_data['log_PulseRate'].max()
#     x_grid = np.linspace(x_min, x_max, 50)
    
#     # Create prediction data
#     pred_data = pd.DataFrame({
#         'log_PulseRate': x_grid,
#         'Cohort': [cohort] * 50,
#         'Animal': [cohort_data['Animal'].iloc[0]] * 50
#     })
    
#     # Get predictions (population average)
#     pred_data['predicted'] = fittedModel_env.predict(pred_data)
    
#     # Plot regression line
#     axes[0, 1].plot(x_grid, pred_data['predicted'], 
#                     color=cohort_colors[idx], linewidth=2.5, 
#                     label=f'{cohort} fit')

# axes[0, 1].set_xlabel('log(PulseRate)')
# axes[0, 1].set_ylabel('VE ENV')
# axes[0, 1].set_title('VE ENV vs log(PulseRate) by Cohort')
# axes[0, 1].legend(loc='best', fontsize='small')
# axes[0, 1].grid(True, alpha=0.3)

# # Plot 3: Cohort slopes comparison for VE PT
# axes[1, 0].set_title('Cohort Slopes for VE PT')
# cohort_slopes_pt = []
# cohort_names_pt = []
# slope_errors_pt = []

# # Get reference cohort slope
# if 'log_PulseRate' in fittedModel_pt.params:
#     ref_slope = fittedModel_pt.params['log_PulseRate']
#     cohort_slopes_pt.append(ref_slope)
#     cohort_names_pt.append(f"Ref: {df_pt['Cohort'].cat.categories[0]}")
    
#     # Calculate SE for reference slope
#     if 'log_PulseRate' in fittedModel_pt.cov_params().index:
#         se = np.sqrt(fittedModel_pt.cov_params().loc['log_PulseRate', 'log_PulseRate'])
#     else:
#         se = fittedModel_pt.bse['log_PulseRate']
#     slope_errors_pt.append(se)

# # Calculate slopes for other cohorts
# for cohort in cohorts[1:]:
#     interaction_term = f'log_PulseRate:C(Cohort)[T.{cohort}]'
#     if interaction_term in fittedModel_pt.params:
#         cohort_slope = ref_slope + fittedModel_pt.params[interaction_term]
#         cohort_slopes_pt.append(cohort_slope)
#         cohort_names_pt.append(str(cohort))
        
#         # Calculate SE for the sum
#         var_a = fittedModel_pt.cov_params().loc['log_PulseRate', 'log_PulseRate']
#         var_b = fittedModel_pt.cov_params().loc[interaction_term, interaction_term]
#         cov_ab = fittedModel_pt.cov_params().loc['log_PulseRate', interaction_term]
#         se = np.sqrt(var_a + var_b + 2*cov_ab)
#         slope_errors_pt.append(se)

# # Plot with error bars
# y_pos = range(len(cohort_slopes_pt))
# axes[1, 0].errorbar(cohort_slopes_pt, y_pos, xerr=[1.96 * se for se in slope_errors_pt],
#                    fmt='o', capsize=5, markersize=8, linewidth=2)
# axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
# axes[1, 0].set_yticks(y_pos)
# axes[1, 0].set_yticklabels(cohort_names_pt)
# axes[1, 0].set_xlabel('Slope Estimate (±95% CI)')
# axes[1, 0].grid(True, alpha=0.3, axis='x')

# # Add slope values
# for i, (slope, se) in enumerate(zip(cohort_slopes_pt, slope_errors_pt)):
#     axes[1, 0].text(slope + 2.5*se, i, f'{slope:.3f}', 
#                    verticalalignment='center', fontsize=9)

# # Plot 4: Cohort slopes comparison for VE ENV
# axes[1, 1].set_title('Cohort Slopes for VE ENV')
# cohort_slopes_env = []
# cohort_names_env = []
# slope_errors_env = []

# # Get reference cohort slope
# if 'log_PulseRate' in fittedModel_env.params:
#     ref_slope = fittedModel_env.params['log_PulseRate']
#     cohort_slopes_env.append(ref_slope)
#     cohort_names_env.append(f"Ref: {df_env['Cohort'].cat.categories[0]}")
    
#     # Calculate SE for reference slope
#     if 'log_PulseRate' in fittedModel_env.cov_params().index:
#         se = np.sqrt(fittedModel_env.cov_params().loc['log_PulseRate', 'log_PulseRate'])
#     else:
#         se = fittedModel_env.bse['log_PulseRate']
#     slope_errors_env.append(se)

# # Calculate slopes for other cohorts
# for cohort in cohorts[1:]:
#     interaction_term = f'log_PulseRate:C(Cohort)[T.{cohort}]'
#     if interaction_term in fittedModel_env.params:
#         cohort_slope = ref_slope + fittedModel_env.params[interaction_term]
#         cohort_slopes_env.append(cohort_slope)
#         cohort_names_env.append(str(cohort))
        
#         # Calculate SE for the sum
#         var_a = fittedModel_env.cov_params().loc['log_PulseRate', 'log_PulseRate']
#         var_b = fittedModel_env.cov_params().loc[interaction_term, interaction_term]
#         cov_ab = fittedModel_env.cov_params().loc['log_PulseRate', interaction_term]
#         se = np.sqrt(var_a + var_b + 2*cov_ab)
#         slope_errors_env.append(se)

# # Plot with error bars
# y_pos = range(len(cohort_slopes_env))
# axes[1, 1].errorbar(cohort_slopes_env, y_pos, xerr=[1.96 * se for se in slope_errors_env],
#                    fmt='o', capsize=5, markersize=8, linewidth=2)
# axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
# axes[1, 1].set_yticks(y_pos)
# axes[1, 1].set_yticklabels(cohort_names_env)
# axes[1, 1].set_xlabel('Slope Estimate (±95% CI)')
# axes[1, 1].grid(True, alpha=0.3, axis='x')

# # Add slope values
# for i, (slope, se) in enumerate(zip(cohort_slopes_env, slope_errors_env)):
#     axes[1, 1].text(slope + 2.5*se, i, f'{slope:.3f}', 
#                    verticalalignment='center', fontsize=9)

# plt.tight_layout()
# plt.show()

# # ============================================================================
# # SUMMARY STATISTICS
# # ============================================================================
# print("\n" + "="*80)
# print("SUMMARY STATISTICS")
# print("="*80)

# # Print slope comparison tables
# print("\nVE PT Cohort Slope Estimates (±95% CI):")
# print("-" * 60)
# for name, slope, se in zip(cohort_names_pt, cohort_slopes_pt, slope_errors_pt):
#     ci_low = slope - 1.96 * se
#     ci_high = slope + 1.96 * se
#     print(f"{name:15s}: {slope:7.3f} [{ci_low:6.3f}, {ci_high:6.3f}]")

# print("\nVE ENV Cohort Slope Estimates (±95% CI):")
# print("-" * 60)
# for name, slope, se in zip(cohort_names_env, cohort_slopes_env, slope_errors_env):
#     ci_low = slope - 1.96 * se
#     ci_high = slope + 1.96 * se
#     print(f"{name:15s}: {slope:7.3f} [{ci_low:6.3f}, {ci_high:6.3f}]")

# # ============================================================================
# # INTERPRETATION
# # ============================================================================
# print("\n" + "="*80)
# print("KEY INTERPRETATIONS")
# print("="*80)

# print("\n1. FOR VE PT (Phase-locking responses):")
# print("   Positive slope: VE PT increases with higher pulse rates")
# print("   Negative slope: VE PT decreases with higher pulse rates")
# print("   Slope near zero: VE PT is relatively constant across pulse rates")

# print("\n2. FOR VE ENV (Envelope-following responses):")
# print("   Positive slope: ENV sensitivity improves with higher pulse rates")
# print("   Negative slope: ENV sensitivity worsens with higher pulse rates")
# print("   Slope near zero: ENV sensitivity is relatively constant")

# print("\n3. COHORT DIFFERENCES:")
# print("   Different slopes between cohorts indicate that the relationship")
# print("   between pulse rate and neural sensitivity differs between groups")

# print("\n4. RANDOM EFFECTS:")
# print("   Random slopes per Animal: Animals vary in how their VE values")
# print("   change with pulse rate, even within the same cohort")

# # ============================================================================
# # SAVE RESULTS
# # ============================================================================
# # Save the fitted models and predictions
# df_pt['predicted_PT'] = fittedModel_pt.predict(df_pt)
# df_env['predicted_ENV'] = fittedModel_env.predict(df_env)

# # Combine results
# df_results = pd.concat([
#     df_pt[['Animal', 'Cohort', 'Pulserate', 'log_PulseRate', 'VE PT', 'predicted_PT']],
#     df_env[['Animal', 'Cohort', 'Pulserate', 'log_PulseRate', 'VE ENV', 'predicted_ENV']]
# ], axis=0)

# df_results.to_csv('VE_models_results.csv', index=False)
# print("\nResults saved to 'VE_models_results.csv'")

# print("\n" + "="*80)
# print("ANALYSIS COMPLETE")
# print("="*80)

# # %%
# #Comparison descriptive statistics 

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# import warnings
# warnings.filterwarnings('ignore')

# # Set style for better visuals
# plt.style.use('seaborn-v0_8-whitegrid')
# sns.set_palette("Set2")

# # Load your data
# df = pd.read_csv(config.ANALYSIS_ROOT+'/MEM_ENVvsPT.csv')
# df.columns = df.columns.str.strip()
# df = df.rename(columns={'Aimal': 'Animal'})

# print("="*80)
# print("VE PT vs VE ENV: COMPREHENSIVE STATISTICAL ANALYSIS")
# print("="*80)

# # ============================================================================
# # 1. DESCRIPTIVE STATISTICS
# # ============================================================================
# print("\n1. DESCRIPTIVE STATISTICS")
# print("-"*40)

# # Calculate basic statistics
# ve_pt_stats = {
#     'Mean': df['VE PT'].mean(),
#     'Std': df['VE PT'].std(),
#     'Median': df['VE PT'].median(),
#     'Min': df['VE PT'].min(),
#     'Max': df['VE PT'].max(),
#     'Q1': df['VE PT'].quantile(0.25),
#     'Q3': df['VE PT'].quantile(0.75),
#     'N': len(df['VE PT'])
# }

# ve_env_stats = {
#     'Mean': df['VE ENV'].mean(),
#     'Std': df['VE ENV'].std(),
#     'Median': df['VE ENV'].median(),
#     'Min': df['VE ENV'].min(),
#     'Max': df['VE ENV'].max(),
#     'Q1': df['VE ENV'].quantile(0.25),
#     'Q3': df['VE ENV'].quantile(0.75),
#     'N': len(df['VE ENV'])
# }

# stats_df = pd.DataFrame({'VE PT': ve_pt_stats, 'VE ENV': ve_env_stats})
# print(stats_df.round(4))

# # ============================================================================
# # 2. VISUAL COMPARISON
# # ============================================================================
# print("\n2. VISUAL COMPARISON")
# print("-"*40)

# fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle('VE PT vs VE ENV: Comprehensive Comparison', fontsize=16, fontweight='bold')

# # Plot 1: Boxplot comparison
# axes[0, 0].boxplot([df['VE PT'], df['VE ENV']], labels=['VE PT', 'VE ENV'])
# axes[0, 0].set_ylabel('VE Value')
# axes[0, 0].set_title('Distribution Comparison')
# axes[0, 0].grid(True, alpha=0.3)

# # Add mean markers
# axes[0, 0].scatter([1, 2], [df['VE PT'].mean(), df['VE ENV'].mean()], 
#                   color='red', s=100, marker='x', label='Mean')
# axes[0, 0].legend()

# # Plot 2: Histogram comparison
# axes[0, 1].hist(df['VE PT'], bins=30, alpha=0.7, density=True, label='VE PT')
# axes[0, 1].hist(df['VE ENV'], bins=30, alpha=0.7, density=True, label='VE ENV')
# axes[0, 1].set_xlabel('VE Value')
# axes[0, 1].set_ylabel('Density')
# axes[0, 1].set_title('Distribution Density')
# axes[0, 1].legend()
# axes[0, 1].grid(True, alpha=0.3)

# # Plot 3: Scatter plot
# axes[0, 2].scatter(df['VE PT'], df['VE ENV'], alpha=0.6, s=30)
# axes[0, 2].plot([0, df['VE PT'].max()], [0, df['VE PT'].max()], 
#                'r--', alpha=0.5, label='Identity line')
# axes[0, 2].set_xlabel('VE PT')
# axes[0, 2].set_ylabel('VE ENV')
# axes[0, 2].set_title('VE PT vs VE ENV (paired observations)')
# axes[0, 2].legend()
# axes[0, 2].grid(True, alpha=0.3)

# # Calculate and display correlation
# correlation = df['VE PT'].corr(df['VE ENV'])
# axes[0, 2].text(0.05, 0.95, f'r = {correlation:.3f}', 
#                transform=axes[0, 2].transAxes,
#                verticalalignment='top',
#                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# # ============================================================================
# # 3. STATISTICAL TESTS (WITHIN OBSERVATIONS)
# # ============================================================================
# print("\n3. STATISTICAL TESTS: VE PT vs VE ENV (Paired Comparisons)")
# print("-"*40)

# # Since VE PT and VE ENV are measured on the same trials, use paired tests
# print("\nA. PAIRED T-TEST (parametric):")
# t_stat_paired, p_val_paired = stats.ttest_rel(df['VE PT'], df['VE ENV'])
# print(f"   t({len(df)-1}) = {t_stat_paired:.4f}, p = {p_val_paired:.4f}")

# # Calculate effect size (Cohen's d for paired samples)
# mean_diff = (df['VE PT'] - df['VE ENV']).mean()
# std_diff = (df['VE PT'] - df['VE ENV']).std()
# cohen_d_paired = mean_diff / std_diff
# print(f"   Mean difference (PT - ENV) = {mean_diff:.4f}")
# print(f"   Standard deviation of differences = {std_diff:.4f}")
# print(f"   Cohen's d (paired) = {cohen_d_paired:.3f}")

# print("\nB. WILCOXON SIGNED-RANK TEST (non-parametric):")
# wilcoxon_stat, wilcoxon_p = stats.wilcoxon(df['VE PT'], df['VE ENV'])
# print(f"   Statistic = {wilcoxon_stat:.4f}, p = {wilcoxon_p:.4f}")

# # Interpret results
# print("\nC. INTERPRETATION:")
# if p_val_paired < 0.05:
#     if mean_diff > 0:
#         print(f"   SIGNIFICANT: VE PT is significantly HIGHER than VE ENV")
#         print(f"   Average PT is {mean_diff:.4f} units higher than ENV")
#     else:
#         print(f"   SIGNIFICANT: VE PT is significantly LOWER than VE ENV")
#         print(f"   Average PT is {abs(mean_diff):.4f} units lower than ENV")
    
#     # Interpret effect size
#     if abs(cohen_d_paired) >= 0.8:
#         effect_size = "LARGE"
#     elif abs(cohen_d_paired) >= 0.5:
#         effect_size = "MEDIUM"
#     elif abs(cohen_d_paired) >= 0.2:
#         effect_size = "SMALL"
#     else:
#         effect_size = "VERY SMALL"
    
#     print(f"   Effect size: {effect_size} (Cohen's d = {cohen_d_paired:.3f})")
# else:
#     print(f"   NOT SIGNIFICANT: No statistical difference between VE PT and VE ENV")

# # ============================================================================
# # 4. ACROSS-ANIMAL ANALYSIS
# # ============================================================================
# print("\n" + "="*80)
# print("4. ACROSS-ANIMAL ANALYSIS")
# print("="*80)

# # Calculate animal-level statistics
# animal_stats = []

# for animal in df['Animal'].unique():
#     animal_data = df[df['Animal'] == animal]
    
#     stats_dict = {
#         'Animal': animal,
#         'Cohort': animal_data['Cohort'].iloc[0],
#         'VE_PT_Mean': animal_data['VE PT'].mean(),
#         'VE_ENV_Mean': animal_data['VE ENV'].mean(),
#         'PT_minus_ENV': animal_data['VE PT'].mean() - animal_data['VE ENV'].mean(),
#         'VE_PT_Std': animal_data['VE PT'].std(),
#         'VE_ENV_Std': animal_data['VE ENV'].std(),
#         'N_Observations': len(animal_data)
#     }
#     animal_stats.append(stats_dict)

# animal_df = pd.DataFrame(animal_stats)

# print("\nANIMAL-LEVEL SUMMARY (first 10 animals):")
# print(animal_df.head(10).round(4).to_string())

# # ============================================================================
# # 5. ACROSS-ANIMAL STATISTICAL TESTS
# # ============================================================================
# print("\n5. ACROSS-ANIMAL STATISTICAL TESTS")
# print("-"*40)

# print("\nA. ONE-SAMPLE T-TEST ON ANIMAL MEAN DIFFERENCES:")
# # Test if the average animal difference is different from 0
# t_stat_animal, p_val_animal = stats.ttest_1samp(animal_df['PT_minus_ENV'], 0)
# print(f"   Number of animals: {len(animal_df)}")
# print(f"   Mean animal difference (PT - ENV) = {animal_df['PT_minus_ENV'].mean():.4f}")
# print(f"   SEM = {animal_df['PT_minus_ENV'].sem():.4f}")
# print(f"   t({len(animal_df)-1}) = {t_stat_animal:.4f}, p = {p_val_animal:.4f}")

# # Calculate 95% confidence interval for animal differences
# ci_low = animal_df['PT_minus_ENV'].mean() - 1.96 * animal_df['PT_minus_ENV'].sem()
# ci_high = animal_df['PT_minus_ENV'].mean() + 1.96 * animal_df['PT_minus_ENV'].sem()
# print(f"   95% CI for animal differences: [{ci_low:.4f}, {ci_high:.4f}]")

# print("\nB. PROPORTION OF ANIMALS WITH PT > ENV:")
# n_pt_higher = sum(animal_df['PT_minus_ENV'] > 0)
# prop_pt_higher = n_pt_higher / len(animal_df)
# print(f"   Animals with PT > ENV: {n_pt_higher}/{len(animal_df)} = {prop_pt_higher:.1%}")

# # Binomial test: Is proportion significantly different from 0.5?
# from scipy.stats import binomtest
# binom_result = binomtest(n_pt_higher, len(animal_df), p=0.5)
# print(f"   Binomial test (vs 50%): p = {binom_result.pvalue:.4f}")

# print("\nC. REPEATED MEASURES ANOVA (across animals):")
# # Prepare data for repeated measures ANOVA
# # We need to reshape to long format for statsmodels
# df_long = pd.melt(df, 
#                   id_vars=['Animal', 'Cohort', 'Pulserate', 'stimDuration'],
#                   value_vars=['VE PT', 'VE ENV'],
#                   var_name='VE_type',
#                   value_name='VE_value')

# df_long['VE_type'] = df_long['VE_type'].str.replace('VE ', '')

# # Use statsmodels for repeated measures ANOVA
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import AnovaRM

# try:
#     # Repeated measures ANOVA
#     anova_rm = AnovaRM(df_long, 'VE_value', 'Animal', within=['VE_type'])
#     rm_results = anova_rm.fit()
#     print(f"   Repeated Measures ANOVA:")
#     print(f"   F({rm_results.df_num}, {rm_results.df_denom}) = {rm_results.anova_table['F Value'][0]:.4f}")
#     print(f"   p = {rm_results.anova_table['Pr > F'][0]:.4f}")
# except:
#     # Fallback: Mixed model approach
#     import statsmodels.formula.api as smf
#     mixed_model = smf.mixedlm("VE_value ~ VE_type", df_long, groups=df_long["Animal"])
#     mixed_result = mixed_model.fit(method='lbfgs', maxiter=1000)
#     print(f"   Mixed Effects Model (alternative):")
#     if 'VE_type[T.PT]' in mixed_result.pvalues:
#         print(f"   VE_type effect: β = {mixed_result.params['VE_type[T.PT]']:.4f}")
#         print(f"   p = {mixed_result.pvalues['VE_type[T.PT]']:.4f}")

# # ============================================================================
# # 6. VISUALIZE ANIMAL-LEVEL DIFFERENCES
# # ============================================================================
# # Plot 4: Animal-level differences
# animal_df_sorted = animal_df.sort_values('PT_minus_ENV')
# axes[1, 0].barh(range(len(animal_df_sorted)), animal_df_sorted['PT_minus_ENV'], 
#                color=['salmon' if x > 0 else 'skyblue' for x in animal_df_sorted['PT_minus_ENV']])
# axes[1, 0].axvline(x=0, color='black', linewidth=1, alpha=0.5)
# axes[1, 0].set_xlabel('PT - ENV Difference')
# axes[1, 0].set_ylabel('Animal (sorted)')
# axes[1, 0].set_title('Animal-Specific Differences')
# axes[1, 0].grid(True, alpha=0.3, axis='x')

# # Add animal labels for first and last few
# for i in [0, 1, len(animal_df_sorted)-2, len(animal_df_sorted)-1]:
#     if i < len(animal_df_sorted):
#         axes[1, 0].text(animal_df_sorted.iloc[i]['PT_minus_ENV'], i, 
#                        animal_df_sorted.iloc[i]['Animal'][:10], 
#                        va='center', fontsize=8,
#                        ha='left' if animal_df_sorted.iloc[i]['PT_minus_ENV'] > 0 else 'right')

# # Plot 5: Distribution of animal differences
# axes[1, 1].hist(animal_df['PT_minus_ENV'], bins=15, edgecolor='black', alpha=0.7)
# axes[1, 1].axvline(x=animal_df['PT_minus_ENV'].mean(), color='red', 
#                   linestyle='--', linewidth=2, label=f'Mean: {animal_df["PT_minus_ENV"].mean():.2f}')
# axes[1, 1].axvline(x=0, color='black', linewidth=1, alpha=0.5)
# axes[1, 1].set_xlabel('PT - ENV Difference')
# axes[1, 1].set_ylabel('Number of Animals')
# axes[1, 1].set_title('Distribution of Animal Differences')
# axes[1, 1].legend()
# axes[1, 1].grid(True, alpha=0.3)

# # Plot 6: Individual animal profiles (top 8 animals with largest differences)
# top_n = min(8, len(animal_df))
# animal_df_largest = pd.concat([
#     animal_df.nlargest(top_n//2, 'PT_minus_ENV'),
#     animal_df.nsmallest(top_n//2, 'PT_minus_ENV')
# ])

# for i, (_, row) in enumerate(animal_df_largest.iterrows()):
#     x_pos = [i*3, i*3 + 1]
#     axes[1, 2].bar(x_pos, [row['VE_PT_Mean'], row['VE_ENV_Mean']], 
#                   color=['salmon', 'skyblue'], alpha=0.7)
#     # Add error bars (SEM)
#     pt_sem = row['VE_PT_Std'] / np.sqrt(row['N_Observations'])
#     env_sem = row['VE_ENV_Std'] / np.sqrt(row['N_Observations'])
#     axes[1, 2].errorbar(x_pos, [row['VE_PT_Mean'], row['VE_ENV_Mean']],
#                        yerr=[pt_sem, env_sem], fmt='none', color='black', 
#                        capsize=3, alpha=0.7)

# axes[1, 2].set_xticks([i*3 + 0.5 for i in range(len(animal_df_largest))])
# axes[1, 2].set_xticklabels([f"{row['Animal'][:8]}..." for _, row in animal_df_largest.iterrows()], 
#                           rotation=45, fontsize=9)
# axes[1, 2].set_ylabel('Mean VE Value (±SEM)')
# axes[1, 2].set_title('Individual Animal Profiles (Extreme Cases)')
# axes[1, 2].grid(True, alpha=0.3, axis='y')

# # Add legend for plot 6
# from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor='salmon', alpha=0.7, label='VE PT'),
#                    Patch(facecolor='skyblue', alpha=0.7, label='VE ENV')]
# axes[1, 2].legend(handles=legend_elements, loc='upper right')

# plt.tight_layout()
# plt.show()

# # ============================================================================
# # 7. COHORT ANALYSIS (if applicable)
# # ============================================================================
# print("\n6. COHORT ANALYSIS")
# print("-"*40)

# if len(df['Cohort'].unique()) > 1:
#     cohorts = sorted(df['Cohort'].unique())
    
#     print(f"\nNumber of cohorts: {len(cohorts)}")
#     print(f"Cohorts: {cohorts}")
    
#     # Calculate cohort-level statistics
#     cohort_results = []
    
#     for cohort in cohorts:
#         cohort_animals = animal_df[animal_df['Cohort'] == cohort]
#         cohort_data = df[df['Cohort'] == cohort]
        
#         cohort_stats = {
#             'Cohort': cohort,
#             'N_Animals': len(cohort_animals),
#             'Mean_PT': cohort_data['VE PT'].mean(),
#             'Mean_ENV': cohort_data['VE ENV'].mean(),
#             'Mean_Difference': cohort_animals['PT_minus_ENV'].mean(),
#             'SEM_Difference': cohort_animals['PT_minus_ENV'].sem(),
#             'Prop_PT_higher': (cohort_animals['PT_minus_ENV'] > 0).mean()
#         }
#         cohort_results.append(cohort_stats)
    
#     cohort_df = pd.DataFrame(cohort_results)
#     print("\nCOHORT-LEVEL STATISTICS:")
#     print(cohort_df.round(4).to_string())
    
#     # Test for cohort differences in PT-ENV differences
#     if len(cohorts) >= 2:
#         print("\nTESTING FOR COHORT DIFFERENCES:")
        
#         # One-way ANOVA on animal differences
#         cohort_groups = [animal_df[animal_df['Cohort'] == cohort]['PT_minus_ENV'].values 
#                         for cohort in cohorts]
#         f_stat, p_val_cohort = stats.f_oneway(*cohort_groups)
        
#         print(f"   One-way ANOVA on animal differences:")
#         print(f"   F({len(cohorts)-1}, {len(animal_df)-len(cohorts)}) = {f_stat:.4f}")
#         print(f"   p = {p_val_cohort:.4f}")
        
#         if p_val_cohort < 0.05:
#             print(f"   → SIGNIFICANT: Cohorts differ in PT-ENV relationship")
            
#             # Post-hoc tests (Tukey if equal variance, Games-Howell if not)
#             from scipy.stats import levene
#             levene_stat, levene_p = levene(*cohort_groups)
            
#             if levene_p > 0.05:  # Equal variance
#                 print(f"   → Equal variances assumed (Levene's p = {levene_p:.4f})")
#                 # Post-hoc t-tests with Bonferroni correction
#                 print(f"   Post-hoc comparisons (Bonferroni-corrected):")
#                 from itertools import combinations
#                 comparisons = list(combinations(cohorts, 2))
#                 alpha_corrected = 0.05 / len(comparisons)
                
#                 for cohort1, cohort2 in comparisons:
#                     group1 = animal_df[animal_df['Cohort'] == cohort1]['PT_minus_ENV']
#                     group2 = animal_df[animal_df['Cohort'] == cohort2]['PT_minus_ENV']
#                     t_stat_post, p_val_post = stats.ttest_ind(group1, group2, equal_var=True)
#                     sig = "**" if p_val_post < alpha_corrected else "*" if p_val_post < 0.05 else ""
#                     print(f"     {cohort1} vs {cohort2}: t = {t_stat_post:.4f}, p = {p_val_post:.4f}{sig}")
#             else:
#                 print(f"   → Unequal variances (Levene's p = {levene_p:.4f})")
#         else:
#             print(f"   → NO SIGNIFICANT DIFFERENCES between cohorts")

# # ============================================================================
# # 8. EFFECT SIZE AND POWER ANALYSIS
# # ============================================================================
# print("\n7. EFFECT SIZE AND POWER ANALYSIS")
# print("-"*40)

# # Calculate various effect sizes
# print("\nA. EFFECT SIZES:")
# print(f"   1. Cohen's d (paired): {cohen_d_paired:.3f}")

# # Glass's delta (using control group SD - here using ENV as reference)
# glass_delta = mean_diff / df['VE ENV'].std()
# print(f"   2. Glass's Δ (using ENV SD): {glass_delta:.3f}")

# # Hedges' g (corrected for small sample bias)
# def hedges_g(mean_diff, std_pooled, n):
#     # Small sample bias correction
#     j = 1 - (3 / (4 * (n - 1) - 1))
#     return (mean_diff / std_pooled) * j

# std_pooled = np.sqrt((df['VE PT'].std()**2 + df['VE ENV'].std()**2) / 2)
# hedges_g_val = hedges_g(mean_diff, std_pooled, len(df))
# print(f"   3. Hedges' g (bias-corrected): {hedges_g_val:.3f}")

# # Common language effect size
# # Probability that a randomly selected PT value > randomly selected ENV value
# from scipy.stats import norm
# cl_es = norm.cdf(mean_diff / np.sqrt(std_diff**2))
# print(f"   4. Common Language ES: {cl_es:.3f}")
# print(f"     Interpretation: {cl_es:.1%} chance that random PT > random ENV")

# print("\nB. POWER ANALYSIS:")
# # Calculate achieved power
# from statsmodels.stats.power import TTestPower
# power_analyzer = TTestPower()
# achieved_power = power_analyzer.power(effect_size=cohen_d_paired, 
#                                      nobs=len(df), 
#                                      alpha=0.05)
# print(f"   Achieved power (α=0.05): {achieved_power:.3f}")

# # Required sample size for 80% power
# required_n = power_analyzer.solve_power(effect_size=cohen_d_paired, 
#                                        power=0.8, 
#                                        alpha=0.05)
# print(f"   Required N for 80% power: {int(np.ceil(required_n))} observations")

# # ============================================================================
# # 9. FINAL SUMMARY AND CONCLUSIONS
# # ============================================================================
# print("\n" + "="*80)
# print("8. FINAL SUMMARY AND CONCLUSIONS")
# print("="*80)

# print(f"\nSTUDY OVERVIEW:")
# print(f"  • Total observations: {len(df)}")
# print(f"  • Number of animals: {len(animal_df)}")
# print(f"  • Measurements per animal: {df.groupby('Animal').size().mean():.1f} ± {df.groupby('Animal').size().std():.1f}")

# print(f"\nMAIN FINDINGS:")
# print(f"  1. VE PT vs VE ENV (all observations):")
# print(f"     • Mean PT: {df['VE PT'].mean():.4f}")
# print(f"     • Mean ENV: {df['VE ENV'].mean():.4f}")
# print(f"     • Difference (PT - ENV): {mean_diff:.4f}")
# print(f"     • Statistical significance: {'p < 0.05' if p_val_paired < 0.05 else 'p > 0.05'}")

# print(f"\n  2. Across-animal analysis:")
# print(f"     • Mean animal difference: {animal_df['PT_minus_ENV'].mean():.4f}")
# print(f"     • SEM: {animal_df['PT_minus_ENV'].sem():.4f}")
# print(f"     • 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
# print(f"     • Animals with PT > ENV: {n_pt_higher}/{len(animal_df)} ({prop_pt_higher:.1%})")

# print(f"\n  3. Effect size:")
# print(f"     • Cohen's d: {cohen_d_paired:.3f}")
# print(f"     • Interpretation: ", end="")
# if abs(cohen_d_paired) >= 0.8:
#     print("LARGE effect")
# elif abs(cohen_d_paired) >= 0.5:
#     print("MEDIUM effect")
# elif abs(cohen_d_paired) >= 0.2:
#     print("SMALL effect")
# else:
#     print("VERY SMALL effect")

# print(f"\n  4. Statistical power:")
# print(f"     • Achieved power: {achieved_power:.1%}")
# if achieved_power >= 0.8:
#     print(f"     • SUFFICIENT power for detection")
# else:
#     print(f"     • INSUFFICIENT power - study may be underpowered")

# print(f"\nCONCLUSION:")
# if p_val_paired < 0.05:
#     if mean_diff > 0:
#         conclusion = f"VE PT is significantly HIGHER than VE ENV by {mean_diff:.4f} units on average."
#     else:
#         conclusion = f"VE PT is significantly LOWER than VE ENV by {abs(mean_diff):.4f} units on average."
    
#     if prop_pt_higher > 0.7:
#         consistency = "highly consistent"
#     elif prop_pt_higher > 0.6:
#         consistency = "moderately consistent"
#     else:
#         consistency = "somewhat inconsistent"
    
#     conclusion += f" This pattern is {consistency} across animals ({prop_pt_higher:.1%} show PT > ENV)."
# else:
#     conclusion = "No statistically significant difference between VE PT and VE ENV was detected."

# print(f"  {conclusion}")

# # ============================================================================
# # 10. SAVE RESULTS
# # ============================================================================
# # Save all results to CSV files
# print("\n" + "="*80)
# print("SAVING RESULTS")
# print("="*80)

# # Save descriptive statistics
# stats_df.to_csv('descriptive_statistics.csv')

# # Save animal-level statistics
# animal_df.to_csv('animal_level_statistics.csv')

# # Save statistical test results
# test_results = {
#     'Test': ['Paired t-test', 'Wilcoxon signed-rank', 'One-sample t-test (animal)'],
#     'Statistic': [t_stat_paired, wilcoxon_stat, t_stat_animal],
#     'p_value': [p_val_paired, wilcoxon_p, p_val_animal],
#     'Effect_size': [cohen_d_paired, np.nan, animal_df['PT_minus_ENV'].mean()/animal_df['PT_minus_ENV'].std()],
#     'N': [len(df), len(df), len(animal_df)]
# }

# test_results_df = pd.DataFrame(test_results)
# test_results_df.to_csv('statistical_test_results.csv', index=False)

# # Save final summary
# summary = {
#     'Metric': ['Total_observations', 'Number_animals', 'Mean_PT', 'Mean_ENV', 
#                'Mean_difference', 'Std_difference', 'Cohen_d', 'p_value',
#                'Prop_animals_PT_higher', 'Achieved_power'],
#     'Value': [len(df), len(animal_df), df['VE PT'].mean(), df['VE ENV'].mean(),
#               mean_diff, std_diff, cohen_d_paired, p_val_paired,
#               prop_pt_higher, achieved_power]
# }

# summary_df = pd.DataFrame(summary)
# summary_df.to_csv('final_summary.csv', index=False)

# print("\nFiles saved:")
# print("1. descriptive_statistics.csv - Basic statistics")
# print("2. animal_level_statistics.csv - Animal-specific analysis")
# print("3. statistical_test_results.csv - All statistical tests")
# print("4. final_summary.csv - Final summary")
# print("\nAnalysis complete!")

# %%

# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# from statsmodels.stats.anova import anova_lm

# # Assuming df is already loaded
# # Check the structure
# df = pd.read_csv(config.ANALYSIS_ROOT+'/MEM_ENVvsPT_New.csv')

# print("Dataframe info:")
# print(df.info())
# print("\nVariable levels:")
# print("stimulusType:", df['stimulusType'].unique())
# print("Pulserate:", df['Pulserate'].unique())
# print("Cohort:", df['Cohort'].unique())
# print("Animal:", df['Animal'].nunique(), "unique animals")

# # Model 1: Basic model testing stimulusType effect
# print("\n--- Model 1: Testing stimulusType main effect ---")
# model1 = smf.mixedlm("VE ~ stimulusType", 
#                      data=df, 
#                      groups=df["Animal"],
#                      re_formula="~1").fit()

# print(model1.summary())

# # Check residuals (optional, for model diagnostics)
# print("\n--- Model diagnostics ---")
# residuals = model1.resid
# print(f"Residuals - Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f}")
# %%


df = pd.read_csv(config.ANALYSIS_ROOT+'/GEE_Model_V3.csv')

print("Anims:", df['Animal'].unique())

print("Dataframe info:")
print(df.info())
print("\nVariable levels:")
print("stimulusType:", df['stimulusType'].unique())
print("Pulserate:", df['Pulserate'].unique())
print("Cohort:", df['Cohort'].unique())
print("Animal:", df['Animal'].unique(), "unique animals")

#Final model 
# Alternative: Generalized Linear Mixed Model if VE is clearly non-normal
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Mixed effects model with GLM for non-normal data
print("--- Generalized Linear Mixed Model for non-normal VE ---")

# Check distribution of VE to inform family choice
print(f"VE distribution stats:\nMean: {df['VE'].mean():.3f}, Std: {df['VE'].std():.3f}")
print(f"Min: {df['VE'].min():.3f}, Max: {df['VE'].max():.3f}")
print(f"Skewness: {stats.skew(df['VE']):.3f}, Kurtosis: {stats.kurtosis(df['VE']):.3f}")

# Based on VE characteristics, choose appropriate family:
# 1. If VE is continuous and positive with right-skew (common in bio data)
print("\nUsing Gamma family (for positive, right-skewed continuous data)")
model = smf.gee("VE ~ C(stimulusType, Treatment('ENV'))", 
                data=df,
                groups=df["Animal"],
                family=sm.families.Gamma(link=sm.families.links.log()),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
# Display results
print("\n" + "="*60)
print("MODEL RESULTS: Effect of stimulusType on VE")
print("="*60)
print(model.summary())
print("\n" + "="*60)

# Key hypothesis test for stimulusType
print("\nPRIMARY HYPOTHESIS TEST:")
print(f"Null: VE(ENV) = VE(PT)")
print(f"Alternative: VE(ENV) ≠ VE(PT)")

# Effect size (if using Gaussian)
coef = model.params['C(stimulusType, Treatment(\'ENV\'))[T.PT]']
se = model.bse['C(stimulusType, Treatment(\'ENV\'))[T.PT]']
print(f"\nEffect size: PT vs ENV")
print(f"Estimate: {coef:.3f} (95% CI: [{coef-1.96*se:.3f}, {coef+1.96*se:.3f}])")

#%% We want to do this type of analysis, but for each Pulserate, Cohort and stimDuration separately
# from scipy.stats import ranksums

p_vals = []

for Cohort in df.Cohort.unique():
    for stimDuration in np.array([0.01 , 0.05, 0.2]):
        for Pulserate in np.array([ 900, 4500]):#df.Pulserate.unique():

            dsub=df.loc[(df.stimDuration==stimDuration) & (df.Pulserate==Pulserate) & (df.Cohort==Cohort)]
            model = smf.gee("VE ~ C(stimulusType, Treatment('ENV'))", 
                            data=dsub,
                            groups=dsub["Animal"],
                            family=sm.families.Gamma(link=sm.families.links.log()),
                            cov_struct=sm.cov_struct.Exchangeable()).fit()    
            p_vals.append(model.pvalues[1])
            # # Assuming df is your DataFrame
            # result = ranksums(dsub[dsub['stimulusType'] == 'PT']['VE'],
            #                   dsub[dsub['stimulusType'] == 'ENV']['VE'])
            # print(f"Statistic: {result.statistic:.3f}, p-value: {result.pvalue:.6f}")

#%% 
model = smf.gee("VE ~ C(stimulusType, Treatment('ENV'))", 
                data=dsub,
                groups=dsub["Animal"],
                family=sm.families.Gamma(link=sm.families.links.log()),
                cov_struct=sm.cov_struct.Exchangeable()).fit()            
print(model.summary())