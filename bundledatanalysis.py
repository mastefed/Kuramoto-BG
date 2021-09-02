import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

path = '/Users/federicom/Desktop/datak12k32.csv'
data = pd.read_csv(path)

data['Val K12'] = round(data['Val K12'], 1)
data['Val K32'] = round(data['Val K32'], 1)

data[['Sync Time Pop1', 'Sync Time Pop2', 'Sync Time Pop3']] = data[['Sync Time Pop1', 'Sync Time Pop2', 'Sync Time Pop3']].fillna(np.inf)


# La riga qui sotto mi crea una tabella pivot in cui si vede la variazione di Sync Pop1 in base a Val K12 e Val K32
valPop1 = pd.pivot_table(data[['Val K12', 'Val K32', 'Sync Time Pop1']], values='Sync Time Pop1', index='Val K12', columns='Val K32')
valPop2 = pd.pivot_table(data[['Val K12', 'Val K32', 'Sync Time Pop2']], values='Sync Time Pop2', index='Val K12', columns='Val K32')
valPop3 = pd.pivot_table(data[['Val K12', 'Val K32', 'Sync Time Pop3']], values='Sync Time Pop3', index='Val K12', columns='Val K32')

val2Pop1 = pd.pivot_table(data[['Val K12', 'Val K32', 'Sync Pop1']], values='Sync Pop1', index='Val K12', columns='Val K32')
val2Pop2 = pd.pivot_table(data[['Val K12', 'Val K32', 'Sync Pop2']], values='Sync Pop2', index='Val K12', columns='Val K32')
val2Pop3 = pd.pivot_table(data[['Val K12', 'Val K32', 'Sync Pop3']], values='Sync Pop3', index='Val K12', columns='Val K32')
val2glob = pd.pivot_table(data[['Val K12', 'Val K32', 'Global Sync']], values='Global Sync', index='Val K12', columns='Val K32')

val3Pop1 = pd.pivot_table(data[['Val K12', 'Val K32', 'Presence of Sub Delta Pop1']], values='Presence of Sub Delta Pop1', index='Val K12', columns='Val K32')
val3Pop2 = pd.pivot_table(data[['Val K12', 'Val K32', 'Presence of Sub Delta Pop2']], values='Presence of Sub Delta Pop2', index='Val K12', columns='Val K32')
val3Pop3 = pd.pivot_table(data[['Val K12', 'Val K32', 'Presence of Sub Delta Pop3']], values='Presence of Sub Delta Pop3', index='Val K12', columns='Val K32')

val4Pop1 = pd.pivot_table(data[['Val K12', 'Val K32', 'Integral of PSD Sync Delta Band Pop1']], values='Integral of PSD Sync Delta Band Pop1', index='Val K12', columns='Val K32')
val4Pop2 = pd.pivot_table(data[['Val K12', 'Val K32', 'Integral of PSD Sync Delta Band Pop2']], values='Integral of PSD Sync Delta Band Pop2', index='Val K12', columns='Val K32')
val4Pop3 = pd.pivot_table(data[['Val K12', 'Val K32', 'Integral of PSD Sync Delta Band Pop3']], values='Integral of PSD Sync Delta Band Pop3', index='Val K12', columns='Val K32')

#################################################################

plt.figure("SyncTimes", figsize=(9,7))

plt.subplot(2,2,1)
plt.title('Sync Time Pop 1')
ax1 = sns.heatmap(valPop1, vmin=1000, vmax=0, cmap='viridis_r', mask=valPop1.isnull())
ax1.invert_yaxis()

plt.subplot(2,2,2)
plt.title('Sync Time Pop 2')
ax1 = sns.heatmap(valPop2, vmin=1000, vmax=0, cmap='viridis_r', mask=valPop2.isnull())
ax1.invert_yaxis()

plt.subplot(2,2,3)
plt.title('Sync Time Pop 3')
ax1 = sns.heatmap(valPop3, vmin=1000, vmax=0, cmap='viridis_r', mask=valPop3.isnull())
ax1.invert_yaxis()

plt.subplots_adjust(wspace=0.5, hspace=0.5)

#################################################################

plt.figure("SyncValues", figsize=(9,7))

plt.subplot(2,2,1)
plt.title('Sync Pop 1')
ax1 = sns.heatmap(val2Pop1, vmin=0, vmax=1, cmap='viridis')
ax1.invert_yaxis()

plt.subplot(2,2,2)
plt.title('Sync Pop 2')
ax1 = sns.heatmap(val2Pop2, vmin=0, vmax=1, cmap='viridis')
ax1.invert_yaxis()

plt.subplot(2,2,3)
plt.title('Sync Pop 3')
ax1 = sns.heatmap(val2Pop3, vmin=0, vmax=1, cmap='viridis')
ax1.invert_yaxis()

plt.subplot(2,2,4)
plt.title('Glob Sync')
ax1 = sns.heatmap(val2glob, vmin=0, vmax=1, cmap='viridis')
ax1.invert_yaxis()

plt.subplots_adjust(wspace=0.5, hspace=0.5)

#################################################################
colors = ["gold", "royalblue"] 
mycmap = LinearSegmentedColormap.from_list('Custom', colors, len(colors))

plt.figure("SubDeltas", figsize=(9,7))

plt.subplot(2,2,1)
plt.title('Sub Delta Presence in Sync PSD of Pop 1')
ax1 = sns.heatmap(val3Pop1, cmap=mycmap, cbar_kws={"shrink": .3})
ax1.invert_yaxis()

colorbar = ax1.collections[0].colorbar
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(['False', 'True'])

plt.subplot(2,2,2)
plt.title('Sub Delta Presence in Sync PSD of Pop 2')
ax1 = sns.heatmap(val3Pop2, cmap=mycmap, cbar_kws={"shrink": .3})
ax1.invert_yaxis()

colorbar = ax1.collections[0].colorbar
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(['False', 'True'])

plt.subplot(2,2,3)
plt.title('Sub Delta Presence in Sync PSD of Pop 3')
ax1 = sns.heatmap(val3Pop3, cmap=mycmap, cbar_kws={"shrink": .3})
ax1.invert_yaxis()

colorbar = ax1.collections[0].colorbar
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(['False', 'True'])

plt.subplots_adjust(wspace=0.5, hspace=0.5)

#################################################################

plt.figure("Deltas", figsize=(9,7))

plt.subplot(2,2,1)
plt.title('Delta Band integral in Sync PSD for Pop 1')
ax1 = sns.heatmap(val4Pop1, vmin=0., vmax=0.006, cmap='viridis', cbar_kws={"shrink": .9})
ax1.invert_yaxis()

plt.subplot(2,2,2)
plt.title('Delta Band integral in Sync PSD for Pop 2')
ax1 = sns.heatmap(val4Pop2, vmin=0., vmax=0.006, cmap='viridis', cbar_kws={"shrink": .9})
ax1.invert_yaxis()

plt.subplot(2,2,3)
plt.title('Delta Band integral in Sync PSD for Pop 3')
ax1 = sns.heatmap(val4Pop3, vmin=0., vmax=0.006, cmap='viridis', cbar_kws={"shrink": .9})
ax1.invert_yaxis()

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()