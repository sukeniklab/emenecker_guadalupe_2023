# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:02:55 2023

@author: Karina Guadalupe
"""
#%% library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cycler
import matplotlib as mpl
from scipy.stats import ttest_ind
import math

# %% importing data 
df = pd.read_csv("GOOSE_master_dataframe.csv")
summary_GOOSE=pd.read_csv("summary_GOOSE_mergedInfo.csv")
#%% defined function for filtering and plotting
def identify_outliers_list(df, column, threshold=1.5):
    # Calculate the IQR
    q1 = np.percentile(df[column], 25)
    q3 = np.percentile(df[column], 75)
    iqr = q3 - q1
    # Calculate the lower and upper bounds
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    # Identify the outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def dropping_outliers(df, col1, col2, values):
    medians_efret = df.groupby([col1, col2])[values].median().reset_index()
    outlyinglist = identify_outliers_list(medians_efret, values)
    outlyinglist = outlyinglist.rename(columns={values: values+'_median'})
    # Merge the two dataframes,Filter the rows that are not present in both dataframes,Drop the indicator column
    merged_df = df.merge(outlyinglist, how='outer', indicator=True)
    filtered_df = merged_df[merged_df['_merge'] != 'both']
    filtered_df.drop(columns='_merge', inplace=True)
    # Remove outliers from medians_efret
    merged_medians_efret = medians_efret.merge(
        outlyinglist, how='outer', indicator=True)
    medians_efret_filtered = merged_medians_efret[merged_medians_efret['_merge'] != 'both']
    medians_efret_filtered.drop(columns='_merge', inplace=True)
    return (pd.DataFrame(filtered_df), pd.DataFrame(medians_efret_filtered))
# median averages and stds using two sample ttest
def ttest(dfx, group_col, data_col):
    groups = dfx[group_col].unique()  # Get unique group values
    results = []  # List to store test results
    # Perform Mann-Whitney U test for pairwise comparisons
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group1 = dfx[dfx[group_col] == groups[i]][[data_col, 'counts']]
            pos1 = int(dfx[dfx[group_col] == groups[i]]['position'].unique())
            group2 = dfx[dfx[group_col] == groups[j]][[data_col, 'counts']]
            pos2 = int(dfx[dfx[group_col] == groups[j]]['position'].unique())

            mean1 = group1[data_col].mean()
            std1 = group1[data_col].std()
            mean2 = group2[data_col].mean()
            std2 = group2[data_col].std()

            ttest_output = ttest_ind(group1[data_col], group2[data_col])
            pvalue = ttest_output[1]

            result = {
                'Group1': groups[i], 'position1': pos1, 'mean1': mean1, 'std1': std1,
                'Group2': groups[j], 'position2': pos2, 'mean2': mean2, 'std2': std2,
                'p-value': pvalue}
            results.append(result)
    return pd.DataFrame(results)

# defining a function for printing pvalue on violin plot
def print_pval(dfz, ax, offset):
    # offset  # Vertical offset for each p-value
    for i, row in dfz.iterrows():
        position1 = row['position1']
        position2 = row['position2']
        mean1 = row['mean1']
        mean2 = row['mean2']
        std1 = row['std1']
        std2 = row['std2']
        
        ax.scatter([position1, position2], [mean1, mean2], marker='s',
                   c='w', s=75, zorder=3, linewidth=10, edgecolor='black')
        pvalue = row['p-value']
        # Check if pvalue is NaN, and skip the row if it is
        if math.isnan(pvalue):  # Use math.isnan() for float data
            continue

        if pvalue < 0.00001:
            starpval = "****"
        elif pvalue < 0.0001:
            starpval = "***"
        elif pvalue < 0.001:
            starpval = "**"
        elif pvalue < 0.01:
            starpval = "*"
        else:
            starpval = "ns"
    
        ax.errorbar([position1, position2], [mean1, mean2], yerr=[
                    std1, std2], fmt='s', color='black', ecolor='black', capsize=20, linewidth=8, zorder=3)
        ax.plot([position1, position2], [(i/21) + offset, (i/21) + offset],
                linewidth=5, color='black')  # Plot the square bracket
        # ax.text((position1 + position2) / 2,(i/11) + 1.06 * offset, f'p-value: {pvalue:.4g}', ha='center',fontsize=45)  # Print p-value above the bracket
        ax.text((position1 + position2) / 2, (i/21) + 1.01 * offset, starpval,
                ha='center', fontsize=85)  # Print p-value above the bracket #1.01 used to be 1.06

#%% paired basal violins - Figure 2-3 function
def paired_basal_outlinedviolin_map(yaxis,sequencelists,figname,figsize):
    fig, ax = plt.subplots(1,len(sequencelists),figsize=figsize, sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.0, wspace=0.1)
    for Idx in range(len(sequencelists)):
        sequencelist=sequencelists[Idx]
        selection = df.loc[df['SequenceNumber'].isin(sequencelist)]
        selection = selection.sort_values(by=['SequenceNumber'])
        seqnumbers = selection.SequenceNumber.unique()
        overallmedians = pd.DataFrame()
        n = len(seqnumbers)
        color = plt.cm.rainbow(np.linspace(0, 1, n+6))
        mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)
        count = 0  # to determine the positions of the violin - should add 24 when it moves to next sequence
        for seqIdx in range(len(seqnumbers)):
            count = count+24
            seq = seqnumbers[seqIdx]
            selectedsequence = selection[selection['SequenceNumber'] == seq]
            sequencemedians = pd.DataFrame()
            # replicate is date and repeats is well. keeping variable name the same
            replicates = selectedsequence.date.unique()
            filtered_df, medians_efret_filtered = dropping_outliers(
                selectedsequence, "dateImaged", "well", yaxis)
            # filtered_df=selectedsequence.copy()
            numberofreplicates = len((filtered_df.groupby(
                ["SequenceNumber", "dateImaged", "well"]).size()))
            for r in range(len(replicates)):
                sliced = filtered_df.loc[filtered_df['date'] == replicates[r]]
                df1 = {k: v.values for k, v in sliced.groupby('well')[
                    yaxis]}
                # Create a list to store the positions of violins
                positions = [count] * len(sliced.well.unique())
                partsh = ax[Idx].violinplot(
                    df1.values(), positions=positions, widths=20, showextrema=False)
                for pc in partsh['bodies']:
                    pc.set_facecolor('none')
                    pc.set_edgecolor('grey')
                    pc.set_lw(6)
                    pc.set_alpha(.2)
                # getting medians of sliced data frame (same date, same seq just different wells)
                wells = sliced.well.unique()
                for w in wells:
                    slicedwell = sliced[sliced['well'] == w]
                    median = slicedwell[yaxis].median()
                    counts = len(slicedwell[yaxis])
                    overallmedians = overallmedians.append({'seq': seq, 'date': replicates[r], 'well': w,
                                                            'median_basal': median, 'position': count, 'counts': counts,
                                                            'n': numberofreplicates}, ignore_index=True)
                    sequencemedians = sequencemedians.append({'seq': seq, 'date': replicates[r], 'well': w,
                                                              'median_basal': median, 'position': count, 'counts': counts,
                                                              'n': numberofreplicates}, ignore_index=True)
            ax[Idx].axhline(y=sequencemedians['median_basal'].mean(),
                          linestyle='--', color='black', lw=4)
            ax[Idx].annotate('seq:'+str(seq)+'\nn = '+str(int(numberofreplicates)), xy=(count, 0.85),
                           xytext=(count, 0.8), fontsize=35, ha='center', va='center')
            ax[Idx].set_ylim(0.85, 0.16)
            ax[Idx].set_xticklabels('')
        results_ttest = ttest(overallmedians, 'seq', 'median_basal')
        print_pval(results_ttest, ax[Idx], 0.75)
    fig.text(0.04, 0.55, yaxis, va='center', rotation='vertical', size=80)
    # plt.savefig(figname+".svg", format="svg",bbox_inches='tight', dpi=1200)
#%% Figure 2 
paired_basal_outlinedviolin_map('Efret_before',[[6, 20],[12,26],[10,24],[18,32]],'Fig_2C',(30, 16))
paired_basal_outlinedviolin_map('Efret_before',[[5,19],[11,25],[9,23],[17,31]],'Fig_2D',(30, 16))
paired_basal_outlinedviolin_map('Efret_before',[[8,22],[14,28],[7,21],[9,23]],'Fig_2E',(60, 16)) ##pair 7,21 don't match Alex's
paired_basal_outlinedviolin_map('Efret_before',[[6,8,10],[12,14,18],[5,7,9],[11,13,17]],'Fig_2F',(60, 16))
paired_basal_outlinedviolin_map('Efret_before',[[20,22,24],[26,28,32]],'Fig_2G',(30, 16))
#%% Figure 3
paired_basal_outlinedviolin_map('Efret_before',[[6,12],[20,26],[19,25],[1,2],[7,13],[10,18],[9,17],[24,32]],'Fig_3C',(115, 18))
paired_basal_outlinedviolin_map('Efret_before',[[5,11],[5,11]],'Fig_3D',(30,16))
paired_basal_outlinedviolin_map('Efret_before',[[21,27],[21,27]],'Fig_3E',(30,16))
paired_basal_outlinedviolin_map('Efret_before',[[8,14],[22,28]],'Fig_3F',(30,16))
#%% Figure 4C
# outlined delta violins - Figure 4B
def delta_outlinedviolin(sequencelists,figname):
    fig, ax = plt.subplots(1,len(sequencelists),figsize=(36, 18),sharex=True,sharey=True)
    for Idx in range(len(sequencelists)):
        sequencenumber=sequencelists[Idx]
        selection = df.loc[df['SequenceNumber'].isin(sequencenumber)]
        seq=str(sequencenumber[0])
        conditions = [100, 300, 750]
        colors = ['blue', 'gray', 'red']
        allweighted_medians = pd.DataFrame()
        for conIdx, con in enumerate(conditions):
            color = colors[conIdx]
            con = conditions[conIdx]
            selectedcon = selection[(selection.condition == con) & (
                abs(selection['ch7']-selection['ch3']) < 2000)]
            filtered_df, medians_efret_filtered = dropping_outliers(
                selectedcon, "dateImaged", "well", "deltaFRET")
            numberofreplicates = len((filtered_df.groupby(
                ["dateImaged", 'condition', "well"]).size()))
            replicates = filtered_df.date.unique()
            for r in range(len(replicates)):
                sliced = filtered_df.loc[filtered_df['date'] == replicates[r]]
                df1 = {k: v.values for k, v in sliced.groupby('well')['deltaFRET']}
                # Create a list to store the positions of violins
                positions = [conIdx*22] * len(sliced.well.unique())
                partsh = ax[Idx].violinplot(
                    df1.values(), positions=positions, widths=20, showextrema=False)
                for pc in partsh['bodies']:
                    pc.set_facecolor('none')
                    pc.set_edgecolor(color)
                    pc.set_lw(4)
                    pc.set_alpha(.3)
    
                wells = sliced.well.unique()
                for well in wells:
                    slicedwell = sliced[sliced['well'] == well]
                    median = slicedwell['deltaFRET'].median()
                    counts = len(slicedwell['deltaFRET'])
                    allweighted_medians = allweighted_medians.append(
                        {'condition': conditions[conIdx], 'date': replicates[r], 'well': well, 'median_delta': median,
                         'counts': counts, 'position': conIdx*22}, ignore_index=True)
                    ax[Idx].axhline(y=0, linestyle='--', color='black', lw=4)
                ax[Idx].annotate('n = '+str(int(numberofreplicates)), xy=(conIdx*22, 0.18),
                                xytext=(conIdx*22, 0.18), fontsize=40, ha='center', va='center')
                ax[Idx].set_ylim(0.22, -0.14)
        ax[Idx].annotate('#'+str(seq), xy=(conIdx*22, -0.12),
                            xytext=(conIdx*22, -0.12), fontsize=50, ha='center', va='center')

        delta_stat_results = ttest(allweighted_medians, 'condition', 'median_delta')
        print_pval(delta_stat_results, ax[Idx], 0.05)
        ax[Idx].set_xticks([0,20,40])
        ax[Idx].set_xticklabels([100,300,750])
    fig.text(0.03, 0.54, '\u2206 $E_f$', va='center', rotation='vertical', size=80)
    # plt.savefig(figname+".svg", format="svg",bbox_inches='tight', dpi=1200)
delta_outlinedviolin([[5],[31],[18]],'Fig_4B')
#%% Figure 4E 
# plotting correlations of features with deltaEf and getting pearson's R 
##and color coding background to make heatmap based on R coeff
features = ['fraction_disorder_promoting','FCR','kappa','basal_mean','avg_re']
nrows = int(np.floor(np.sqrt(len(features))))
ncols = int(np.ceil(len(features)/nrows))
cmap = plt.get_cmap('Reds')

fig, ax = plt.subplots(nrows, ncols,figsize=(35, 25),sharex=True,sharey=False)
fig.subplots_adjust(hspace=0.2, wspace=0.5)
# Create an empty list to store the colors used for the background
background_colors = []
for Idx in range(len(features)):
    feature=features[Idx]
    col = (Idx % ncols)
    row = int(np.floor((Idx)/ncols))
   # Exclude rows with NaN values in the 'feature' column
    valid_rows = ~np.isnan(summary_GOOSE[feature])
    valid_rows = valid_rows & ~summary_GOOSE['SequenceNumber'].isin([6,7,10,19,23,24])
    if feature == 'deltaFRET_mean_100':
        excludedSeqs_hypo=[10,19,23,24]#,#not enough repeats after filtering out violons n>60 cells
                            # 13, 25, 27, 1, 28, 10, 19, 24] #originally excluded
        valid_rows = valid_rows & ~summary_GOOSE['SequenceNumber'].isin(excludedSeqs_hypo)
    elif feature == 'basal_mean':
        # excludedSeqs_hyper=[6,7,10,19,23,24] #not enough repeats after filtering out violons n>60 cells
        # valid_rows = valid_rows & ~summary_GOOSE['SequenceNumber'].isin(excludedSeqs_hyper)
        ax[row,col].set_ylim(0.49,0.31)
    elif feature == 'avg_re' or feature == 'avg_rg':
        excludedSeqs_simulations=[9,10,17,18,24,32] #outliers for linear fit ree to basal Ef #positively charged seqs
        valid_rows = valid_rows & ~summary_GOOSE['SequenceNumber'].isin(excludedSeqs_simulations)
    x = summary_GOOSE['deltaFRET_mean_750'][valid_rows]
    y = summary_GOOSE[feature][valid_rows]
    x_min = -0.026
    x_max = 0.051
    # Linear fits
    x_array1 = np.linspace(x_min,x_max, len(valid_rows))
    ax[row, col].scatter(x, y,
                         s=800, alpha=0.6, edgecolors='black', linewidths=3, c='grey')
    #linear fits 
    fit1,cov1=np.polyfit(x,y,1,cov=True)
    fit_err1=np.sqrt(np.diag(cov1))
    y_fit1=x_array1*fit1[0]+fit1[1]
    y_fit_top1=x_array1*(fit1[0]+fit_err1[0])+(fit1[1]+fit_err1[1])
    y_fit_bott1=x_array1*(fit1[0]-fit_err1[0])+(fit1[1]-fit_err1[1])
    # # Calculate Pearson's correlation coefficient (r) between x-axis and feature
    correlation_coefficient = np.corrcoef(x, y)[0, 1]
    # Calculate R-squared (R^2) for the linear regression
    r2 = correlation_coefficient**2
    color = cmap(r2)
    ax[row,col].plot(x_array1,y_fit1,'--',c='black',zorder=2,lw=10,label=f'$R^2$={r2:.2g}')
    # Set the background color of the subplot based on correlation strength
    ax[row, col].set_facecolor(color)
    ax[row,col].set_title(feature,size=50)
    ax[row,col].set_xlim(x_max,x_min)
    for i in [0,1,2]:
        ax[0,i].legend(loc='lower right',fontsize=50,frameon=False)
    for i in [0,1]:
        ax[1,i].legend(loc='lower left',fontsize=50,frameon=False)
    # Create a colorbar for the background colors
colorbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position as needed
colorbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=colorbar_ax)
colorbar.set_label('Correlation Strength', fontsize=80)
fig.text(0.45, 0.01, '\u2206$E_f^{hyper}$', va='center', rotation='horizontal', size=160)
fig.text(0.01, 0.54, 'Features', va='center', rotation='vertical', size=160)
# plt.savefig("small_correlation_quilt.svg", format="svg",bbox_inches='tight') 
#%% Figure 4F and 4G
#filtering out pionts with bad violin plots for each condition
summary_GOOSE_hypo = summary_GOOSE.loc[~summary_GOOSE['SequenceNumber'].isin(
    [13, 25, 27, 1, 28, 10, 19, 24])]
summary_GOOSE_hyper = summary_GOOSE.loc[~summary_GOOSE['SequenceNumber'].isin(
    [13, 10, 19, 23, 24])]
#linear fits 
x_array_hypo=np.linspace(0.3,0.52,len(summary_GOOSE_hypo['basal_mean']))
fit_hypo,cov_hypo=np.polyfit(summary_GOOSE_hypo['basal_mean'], summary_GOOSE_hypo['deltaFRET_mean_100'],1,cov=True)
fit_err_hypo=np.sqrt(np.diag(cov_hypo))
y_fit_hypo=x_array_hypo*fit_hypo[0]+fit_hypo[1]
y_fit_top_hypo=x_array_hypo*(fit_hypo[0]+fit_err_hypo[0])+(fit_hypo[1]+fit_err_hypo[1])
y_fit_bott_hypo=x_array_hypo*(fit_hypo[0]-fit_err_hypo[0])+(fit_hypo[1]-fit_err_hypo[1])

x_array_hyper=np.linspace(0.3,0.52,len(summary_GOOSE_hyper['basal_mean']))
fit_hyper,cov_hyper=np.polyfit(summary_GOOSE_hyper['basal_mean'], summary_GOOSE_hyper['deltaFRET_mean_750'],1,cov=True)
fit_err_hyper=np.sqrt(np.diag(cov_hyper))
y_fit_hyper=x_array_hyper*fit_hyper[0]+fit_hyper[1]
y_fit_top_hyper=x_array_hyper*(fit_hyper[0]+fit_err_hyper[0])+(fit_hyper[1]+fit_err_hyper[1])
y_fit_bott_hyper=x_array_hyper*(fit_hyper[0]-fit_err_hyper[0])+(fit_hyper[1]-fit_err_hyper[1])

# plotting
fig, ax = plt.subplots(1,2,figsize=(20, 10),sharex=True,sharey=True)
fig.subplots_adjust(hspace=0.0, wspace=0.1)
ax[0].scatter(summary_GOOSE_hypo['basal_mean'], summary_GOOSE_hypo['deltaFRET_mean_100'],
            s=800, alpha=0.6, edgecolors='black', linewidths=3, c="gray")
ax[0].errorbar(summary_GOOSE_hypo['basal_mean'], summary_GOOSE_hypo['deltaFRET_mean_100'],
            yerr=summary_GOOSE_hypo['deltaFRET_std_100'], fmt='s', color='black', ecolor='black', capsize=20, linewidth=4, zorder=2)
ax[0].errorbar(summary_GOOSE_hypo['basal_mean'], summary_GOOSE_hypo['deltaFRET_mean_100'],
            xerr=summary_GOOSE_hypo['basal_std'], fmt='s', color='black', ecolor='black', capsize=20, linewidth=4, zorder=2)
ax[0].plot(x_array_hypo,y_fit_hypo,'-',c='blue',zorder=2,lw=12,label='Hypo-osmotic')
ax[0].fill_between(x_array_hypo,y_fit_top_hypo,y_fit_bott_hypo,color='grey',alpha=0.2,zorder=2)
ax[0].axhline(y=0, linestyle='--', color='black', lw=6)
ax[0].legend(loc='lower left',fontsize=35, frameon=False)

ax[1].scatter(summary_GOOSE_hyper['basal_mean'], summary_GOOSE_hyper['deltaFRET_mean_750'],
            s=800, alpha=0.6, edgecolors='black', linewidths=3, c="gray")
ax[1].errorbar(summary_GOOSE_hyper['basal_mean'], summary_GOOSE_hyper['deltaFRET_mean_750'],
            yerr=summary_GOOSE_hyper['deltaFRET_std_750'], fmt='s', color='black', ecolor='black', capsize=20, linewidth=4, zorder=2)
ax[1].errorbar(summary_GOOSE_hyper['basal_mean'], summary_GOOSE_hyper['deltaFRET_mean_750'],
            xerr=summary_GOOSE_hyper['basal_std'], fmt='s', color='black', ecolor='black', capsize=20, linewidth=4, zorder=2)
ax[1].plot(x_array_hyper,y_fit_hyper,'-',c='red',zorder=2,lw=12,label='Hyper-osmotic')
ax[1].fill_between(x_array_hyper,y_fit_top_hyper,y_fit_bott_hyper,color='grey',alpha=0.2,zorder=2)
ax[1].axhline(y=0, linestyle='--', color='black', lw=6)
ax[1].legend(loc='lower left',fontsize=35, frameon=False)
plt.ylim(0.10,-0.06)
plt.xlim(0.3,0.52)
fig.text(-0.03, 0.5, '\u2206$E_f$', va='center',rotation='vertical', size=80)
fig.text(0.4, -0.08, '$E_f^{basal}$', va='center',rotation='horizontal', size=80)
# plt.savefig("Fig_4F_4G.svg", format="svg",bbox_inches='tight', dpi=1200)
