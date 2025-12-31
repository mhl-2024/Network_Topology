# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticks
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy import stats
import seaborn as sns
import pingouin as pg
import os
from matplotlib.patches import Patch 
# %%


plt.rc('font', family='Arial')

CONFIG = {
    'font_size': 33,
    'scatter_color': '#E07B54',
    'scatter_size': 230,
    'box_width': 3,
    'cbar_fraction': 0.04,
    'cbar_pad': 0.01
}

def significance_stars(p):
    """Return significance stars for p-values."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''



def plot_map(ax, legend_label, gdf, field, cmap_name, vmax, vmin, label):
    
    size0 = CONFIG['font_size']
    plt.rc('font', family='Arial', size=size0)
    base_cmap = cm.get_cmap(cmap_name, 512)
    cmap = ListedColormap(base_cmap(np.linspace(0, 1, 256)))

    gdf.plot(ax=ax, column=field, cmap=cmap, vmax=vmax, vmin=vmin,
             edgecolor="black", linewidth=0.1, zorder=2)
    cbar = plt.colorbar(ax.get_children()[0], ax=ax, location='top',
                        fraction=CONFIG['cbar_fraction'], pad=CONFIG['cbar_pad'], shrink=0.5, extend='both')
    cbar.ax.tick_params(length=8, pad=0.08, width=2, labelsize=size0)
    cbar.set_label(label, labelpad=13, fontsize=size0)
    cbar.outline.set_linewidth(2.5)

    # Custom ticks
    tick_dict = {
        'Tk_c': [1.7, 2.2, 2.7],
        'logAI': [-1, -0.5, 0],
        'SR': [0.38, 0.48, 0.58],
        'Slope': [0.02, 0.08, 0.14]
    }
    if field in tick_dict:
        cbar.set_ticks(tick_dict[field])

    ax.axis('off')
    ax.text(0.07, 1.22, legend_label, transform=ax.transAxes,
            fontsize=size0 + 8, fontweight='bold', ha='left', va='top', color='black')


def bindata_mean_scatter_plot(ax1,fig_legend,df,classname,yname,n_bins):
   
    p1 = df[classname].quantile(0.01)
    p99 = df[classname].quantile(0.99)
    interval = (p99 - p1) / n_bins
    bins = np.arange(p1, p99, interval).tolist()
    bin_edges = [df[classname].min()] + bins + [df[classname].max()]
    
    df[classname+'_bins'] = pd.cut(df[classname], bins=bin_edges, include_lowest=True, duplicates="drop")
    # Group by the bins and calculate the mean for each group
    dftmp = df.groupby(classname+'_bins').agg({classname: ['mean','count','std'], yname: ['mean','std']}).reset_index()
    dfplot = pd.DataFrame({
        classname: dftmp[classname]['mean'],
        yname: dftmp[yname]['mean'],
        yname + '_std': dftmp[yname]['std'],
        'count': dftmp[classname]['count']
    })
    
    yerrname=yname+'_err'
    dfplot[yerrname]=dftmp[yname]['std']/np.sqrt(dftmp[classname]['count'])
    xerrname = classname+'_err'
    dfplot[xerrname]=dftmp[classname]['std']/np.sqrt(dftmp[classname]['count'])
    
    size0=33 
    dotColor = '#E07B54'
    s_size = 230
    scatter_a = ax1.scatter(x=dfplot[classname], y=dfplot[yname], c=dotColor, s=s_size, edgecolors='black',alpha=1)
    dfplot.plot(classname,yname,xerr=xerrname,yerr=yerrname, alpha=1,ax=ax1,ls='none',label=None,legend=False,zorder=0,ecolor='#999999',elinewidth=3,capsize=5)
    
    ax1.set_ylim(1.5, 2.9)
    if classname=='logAI':
       ax1.set_xlabel(r'$\log_{10}[\overline{\mathrm{AI}}]$',fontsize=size0+2,)   
       ax1.set_ylabel('Tokunaga \nparameter c', fontsize=size0+2, labelpad=4) 
       p_x = 0.23
    elif classname=='Slope':
       ax1.set_xlabel(r'$\overline{\mathrm{S}}$',fontsize=size0+2,)
       ax1.set_xticks([0.1,0.3])
       ax1.tick_params(axis='y', which='both', left=False, labelleft=False) 
       p_x = 0.23
    elif classname=='SR':
       ax1.set_xticks([0.3,0.5,0.7])
       ax1.set_xlabel(r'$\overline{\mathrm{SR}}$',fontsize=size0+2,)
       ax1.tick_params(axis='y', which='both', left=False, labelleft=False)
       p_x = 0.6
    
    # ---- spearman correlation ------
    df_clean = df[[classname, yname]].dropna()
    spearman_corr, p_value = stats.spearmanr(df_clean[classname], df_clean[yname])
    
    rho_stars = significance_stars(p_value)
    rho_text = f"ρ={spearman_corr:.2f}"
    # t_rho = ax1.text(x=0.23, y=0.12, s=rho_text,
    #                      fontsize=size0, color='black',horizontalalignment='center', verticalalignment='center',
    #                      transform=ax1.transAxes)
    
    t_rho = ax1.text(x=p_x, y=0.86, s=rho_text,
                         fontsize=size0, color='black',horizontalalignment='center', verticalalignment='center',
                         transform=ax1.transAxes)
    
    fig = ax1.get_figure()
    fig.canvas.draw()
    bbox = t_rho.get_window_extent(renderer=fig.canvas.get_renderer())

    inv = ax1.transAxes.inverted()
    bbox_axes = inv.transform([[bbox.x1, bbox.y1]])
    star_x = bbox_axes[0][0] - 0.02  # Slightly to the right
    #star_y = 0.13  # Slightly above
    star_y = 0.87  # Slightly above

    ax1.text(x=star_x, y=star_y, s=rho_stars,
         fontsize=size0-3, fontweight='bold',
         color='black', transform=ax1.transAxes)
       
    boxwidth=3
    ax1.tick_params(labelsize=size0,direction='in',width=1,length=8,pad=10)
    ax1.tick_params(which='minor',direction='in')
    for spine in ['bottom', 'left', 'right', 'top']:
        ax1.spines[spine].set_linewidth(boxwidth)
    
    ax1.text(0.01, 1.2, fig_legend, transform=ax1.transAxes, fontsize=size0+8, fontweight='bold',ha='left', va='top', 
                 color='black')

def classify_quantile(df, col, labels=None):
    """divide dataset"""
    q20, q50 = df[col].quantile([0.2, 0.5])
    if labels is None:
        labels = [
            f'{col} ≤ {q20:.2f}',
            f'{q20:.2f} < {col} ≤ {q50:.2f}',
            f'{col} > {q50:.2f}'
        ]
    df[f'{col}_class'] = pd.cut(
        df[col],
        bins=[-np.inf, q20, q50, np.inf],
        labels=labels
    )
    return df

def bin_stats(df, angle_col, bin_col='Tk_bin'):
    """calculate statistics for each group"""
    return df.groupby(bin_col).agg(
        Tk_c_mean=('Tk_c', 'mean'),
        angle_mean=(angle_col, 'mean'),
        angle_sem=(angle_col, lambda x: x.std(ddof=1) / np.sqrt(len(x)))
    ).reset_index()

def custom_bin_and_plot(df, class_col,x_range,y_range, ax,palette):
    markers = ['o', 'D', '^'] 
    marksize=[20,18,21]
    
    # markers = ['o', 'D', '^'] 
    # marksize=[20,18,21]
    
    markers = ['v', 'd', 'H'] 
    marksize=[22,23,22]

    
    for i, (cls, group) in enumerate(df.groupby(class_col)):
        group = group.copy()

        # Get quantiles
        q01 = group['Tk_c'].quantile(0.01)
        q99 = group['Tk_c'].quantile(0.99)

        # 18 equal-width bins between q01 and q99
        mid_bins = np.linspace(q01, q99, 9)  # 10 intervals = 11 edges
        bin_edges = np.concatenate(([-np.inf], [q01], mid_bins[1:-1], [q99], [np.inf]))

        # Assign bins
        group['Tk_c_bin'] = pd.cut(group['Tk_c'], bins=bin_edges, labels=False)

         # Group by bins and compute mean and SEM
        binned = group.groupby('Tk_c_bin').agg(
            Tk_c_mean=('Tk_c', 'median'),
            angle_mean=('angle', 'median'),
            angle_sem=('angle', lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        ).reset_index()

        # Plot with error bars
        ax.errorbar(binned['Tk_c_mean'], binned['angle_mean'], yerr=binned['angle_sem'],
                    fmt=markers[i], markersize=marksize[i], capsize=5,ecolor='gray',markeredgecolor='black', 
                    label=f'{cls}', color=palette[i])
        
        boxwidth=3
        ax.tick_params(labelsize=36,direction='in',width=1,length=8,pad=10)
        ax.tick_params(which='minor',direction='in')
        for spine in ['bottom', 'left', 'right', 'top']:
            ax.spines[spine].set_linewidth(boxwidth)
        
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)

def ecdfplot(dfplot, classname, num_classes, bin_values, plotfield):
    dfplot=dfplot.rename(columns={'Tk_c':'c'})
    if classname == 'Tk_c':
        classname='c'
    
    dfplot[classname+'_class'] = pd.cut(dfplot[classname], bins=bin_values)
    fig = plt.figure(figsize=(8, 6))
    font = {'family': 'Arial', 'size': 26}
    plt.rc('font', size=26)
    ax = fig.add_subplot(111)
    
    crest_palette = sns.color_palette("rocket_r", n_colors=num_classes)
    start_color = 0 
    end_color = 1  
    class_palette = sns.color_palette(crest_palette.as_hex()[int(start_color * num_classes):int(end_color * num_classes)], n_colors=num_classes)
    
    ecdf_plot = sns.ecdfplot(data=dfplot, x=plotfield,ax=ax,hue=classname+'_class', linewidth=3,palette=class_palette,zorder=1)
    
    # Plot mean 
    class_means = dfplot.groupby(classname+'_class')[plotfield].mean().reset_index()
    ecdf_y=[]
    
    for x in class_means[plotfield]:
        dftmp = dfplot[dfplot[classname+'_class']==class_means[class_means[plotfield]==x][classname+'_class'].iloc[0]]
        ytmp=dftmp[dftmp[plotfield]<=x].shape[0]/dftmp.shape[0]
        ecdf_y.append(ytmp)
    class_means['ecdf']=ecdf_y
    class_means.plot.scatter(plotfield, 'ecdf', marker='o',c=class_palette,edgecolors='black', s=150,ax=ax,zorder=2)
    
    if plotfield=='Bifurcation':
        plotfield='Mean bifurcation angle[°]'
    elif plotfield=='Sidebranch':
        plotfield='Mean side-branching angle[°]'
        ax.legend_.remove()
      
    ax.tick_params(labelsize=29,direction='in',width=1,length=8,pad=10)
    ax.tick_params(which='minor',direction='in')
    for spine in ['bottom', 'left', 'right', 'top']:
        ax.spines[spine].set_linewidth(3)
        
    ax.yaxis.set_major_formatter(mticks.FormatStrFormatter('%.2f'))
    
    ax.set_xlabel(plotfield, fontsize=29)
    ax.set_ylabel('Proportion', fontsize=29)
    ax.tick_params(labelsize=29)
    
# ------------------------------
# Main plotting
# ------------------------------
def main():
    # -----------Load data------------
    
    infolder = 'inputfolder'
    hex10000 = gpd.read_file(os.path.join(infolder,'hex10000.gpkg'))
    Tk =  pd.read_csv(os.path.join(infolder,'Network_attributes.csv'))    
    Tk_angles = pd.read_csv(os.path.join(infolder,'Classified_angles_vs_Tk_c.csv'))
    grouped_by_id0=pd.read_csv(os.path.join(infolder,'basin_aveDeltaHS_junction_l5.csv'))
    jundf = pd.read_csv(os.path.join(infolder,'JunctionAngles.csv'))
    
    # ---- process data -----
    counts = (
        jundf
        .groupby(['outletNHDid', 'deltaHS'])['angle']
        .count()
        .unstack(fill_value=0)
    )
    counts = counts.rename(columns=lambda x: f'count{x}')
    counts['countall'] = counts.sum(axis=1)

    for i in range(5):
        counts[f'frac{i}'] = counts[f'count{i}'] / counts['countall']

    jun_group = (
        counts
        .reset_index()
        .merge(Tk, on='outletNHDid', how='inner')
    )

    # ************* figure 2 *************************************
    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(4, 10, figure=fig, wspace=0.01, hspace=0.18,
                           height_ratios=[1, 0.01, 1, 0.65],
                           width_ratios=[0.9, 1, 1, 0.08, 1, 1, 0.08, 1, 1, 0.3])

    # Map plotting
    fields = [
        ('a', 'Tk_c', 'viridis', 0, 0, 0.90, 0.02, 'Tokunaga parameter c'),
        ('b', 'logAI', 'viridis', 0, 5, 0.95, 0.01, 'log$_{10}$[AI]'),
        ('c', 'SR', 'viridis_r', 2, 0, 0.96, 0.05, 'Slope ratio'),
        ('d', 'Slope', 'viridis_r', 2, 5, 0.95, 0.20, 'Channel slope')
    ]
    
    for fig_lab, field, cmap, row, col, vmax_q, vmin_q, label in fields:
        ax = fig.add_subplot(gs[row, col:col+5])
        vmin_val = hex10000[field].quantile(vmin_q)
        vmax_val = hex10000[field].quantile(vmax_q)
        plot_map(ax, fig_lab, hex10000, field, cmap, vmax_val, vmin_val, label)
    
    ax1 = fig.add_subplot(gs[3, 1:3])
    bindata_mean_scatter_plot(ax1,'e',Tk,'logAI','Tk_c',14)
    ax2 = fig.add_subplot(gs[3, 4:6])
    bindata_mean_scatter_plot(ax2,'f',Tk,'SR','Tk_c',14)
    ax3 = fig.add_subplot(gs[3, 7:9])
    bindata_mean_scatter_plot(ax3,'g',Tk,'Slope','Tk_c',14)
    
  
    y = -0.23
    ax2.xaxis.set_label_coords(0.5, y)  # SR
    ax3.xaxis.set_label_coords(0.5, y)  # S
    ax1.xaxis.set_label_coords(0.5, y + 0.035)
    
    # ****************** figure 3 ********************************
    # process data 
    p1 = jun_group['Tk_c'].quantile(0.01)
    p99 = jun_group['Tk_c'].quantile(0.99)

    middle_edges = np.linspace(p1, p99, 9)
    bin_edges = np.concatenate((
        [-np.inf],
        middle_edges,
        [np.inf]
    ))

    jun_group['Tk_bin'] = pd.cut(
        jun_group['Tk_c'],
        bins=bin_edges,
        include_lowest=True
    )

    bin_stats1 = (
        jun_group
        .groupby('Tk_bin', observed=True)
        .agg(
            Tk_c_mean=('Tk_c', 'mean'),
            frac0_mean=('frac0', 'mean'),
            frac1_mean=('frac1', 'mean'),
            frac2_mean=('frac2', 'mean'),
            frac3_mean=('frac3', 'mean'),
            frac4_mean=('frac4', 'mean')
        )
        .reset_index()
    )
    bin_stats1 = bin_stats1.sort_values('Tk_c_mean')
    
    
    net_deltaHS_mean = (
        jundf
        .groupby(['outletNHDid', 'deltaHS'], observed=True)
        .agg(mean_angle=('angle', 'mean'))
        .reset_index()
    )

    classes = [0, 1, 2, 3, 4]
    data = [
        net_deltaHS_mean.loc[net_deltaHS_mean['deltaHS'] == c, 'mean_angle'].dropna()
        for c in classes
    ]
    print(data)
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.1, wspace=0.15)
    size0 = 36
    boxwidth = 3
    palette3 = sns.color_palette("Purples", n_colors=3)
    blues = cm.Blues(np.linspace(0.1, 0.95, 5))

    # panel a: mean angle vs Tk_c
    dfplot1 = Tk_angles[Tk_angles['angle_type'] == 'sidebranch'].copy()
    dfplot2 = Tk_angles[Tk_angles['angle_type'] == 'bifurcation'].copy()

    q01, q99 = Tk['Tk_c'].quantile([0.01, 0.99])
    mid_bins = np.linspace(q01, q99, 9)
    bin_edges = np.concatenate(([-np.inf], [q01], mid_bins[1:-1], [q99], [np.inf]))
    Tk['Tk_bin'] = pd.cut(Tk['Tk_c'], bins=bin_edges, labels=False)

    dfplot2['Tk_bin'] = pd.cut(dfplot2['Tk_c'], bins=bin_edges, labels=False)
    dfplot1['Tk_bin'] = pd.cut(dfplot1['Tk_c'], bins=bin_edges, labels=False)

    binned_big = bin_stats(dfplot2, 'angle')
    binned_small = bin_stats(dfplot1, 'angle')
    overall_angle = bin_stats(Tk, 'Branching')

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(overall_angle['Tk_c_mean'], overall_angle['angle_mean'], 
                 yerr=overall_angle['angle_sem'], fmt='^', markersize=21, 
                 color=palette3[1], markeredgecolor='black', label='Branching', capsize=4)
    ax1.errorbar(binned_big['Tk_c_mean'], binned_big['angle_mean'], 
                 yerr=binned_big['angle_sem'], fmt='o', markersize=20, 
                 color=palette3[0], markeredgecolor='black', label='Bifurcation', capsize=4)
    ax1.errorbar(binned_small['Tk_c_mean'], binned_small['angle_mean'], 
                 yerr=binned_small['angle_sem'], fmt='D', markersize=18, 
                 color=palette3[2], markeredgecolor='black', label='Side-branching', capsize=4)
    ax1.set_xlim(0.9, 4.6)
    ax1.tick_params(labelbottom=False) 
    ax1.set_ylabel('Mean angle [°]', fontsize=36,labelpad=19)
    ax1.tick_params(labelsize=size0, direction='in', width=1, length=8, pad=10)
    ax1.spines['bottom'].set_linewidth(boxwidth)
    ax1.spines['left'].set_linewidth(boxwidth)
    ax1.spines['right'].set_linewidth(boxwidth)
    ax1.spines['top'].set_linewidth(boxwidth)

    # panel c: slope ratio vs k
    palette = {str(i): blues[i] for i in range(5)}
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(data=grouped_by_id0, x='deltaHS', y='mean_SR', palette=palette,
                width=0.3, showfliers=False, linewidth=2.5, ax=ax2)
    ax2.set_ylabel('Slope ratio', fontsize=36)
    ax2.set_xlabel('')

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_yticks([0, 0.5, 1])
    ax2.tick_params(labelsize=size0, direction='in', width=1, length=8, pad=10)
    ax2.spines['bottom'].set_linewidth(boxwidth)
    ax2.spines['left'].set_linewidth(boxwidth)
    ax2.spines['right'].set_linewidth(boxwidth)
    ax2.spines['top'].set_linewidth(boxwidth)
    ax2.tick_params(labelbottom=False) 

    # panel b: fraction of junctions vs Tk_c 
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    x = bin_stats1['Tk_c_mean']
    ys = [bin_stats1[f'frac{i}_mean'].values for i in range(5)]
    ys_rev = ys[::-1]
    blues_rev = blues[::-1]
    k_labels = list(range(5))[::-1]   # [4,3,2,1,0]

    x_dense = np.linspace(0.9, 4.6, 50)
    from scipy.interpolate import interp1d
    ys_dense = np.zeros((len(ys_rev), len(x_dense)))
    for i, y in enumerate(ys_rev):
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        ys_dense[i] = f(x_dense)

    # --- stacked area ---
    ax3.stackplot(
        x_dense,
        ys_dense,
        colors=blues_rev,
        alpha=0.85,
        edgecolor='none'
    )

    cum = np.zeros_like(x)
    x_text = np.median(x)

    for y, c, k in zip(ys_rev, blues_rev, k_labels):
        mid = cum + y / 2.0
        y_text = np.interp(x_text, x, mid)
        ax3.text(
            x_text, y_text,
            f'k = {k}',
            fontsize=34,
            ha='center',
            va='center',
            color='k'
        )
        cum += y

    ax3.set_xlabel('Tokunaga parameter c', fontsize=36)
    ax3.set_ylabel('Fraction of junctions', fontsize=36)
    ax3.set_ylim(0, 1.0)
    ax3.set_yticks([0, 0.5, 1])
    ax3.set_xticks([1, 2, 3,4])
    ax3.tick_params(
        labelsize=size0,
        direction='in',
        width=1,
        length=8,
        pad=10
    )

    for spine in ax3.spines.values():
        spine.set_linewidth(boxwidth)

    # panel d: Mean angle vs k 
    ax4 = fig.add_subplot(gs[1, 1])
    bplot = ax4.boxplot(data, widths=0.3, patch_artist=True,
                        medianprops=dict(color='k', linewidth=2.5),
                        whiskerprops=dict(color='k', linewidth=2.5),
                        capprops=dict(color='k', linewidth=2.5),
                        boxprops=dict(linewidth=2.5, edgecolor='k'),
                        showfliers=False)
    for patch, color in zip(bplot['boxes'], blues):
        patch.set_facecolor(color)
        patch.set_edgecolor('k')

    ax4.set_xticks(range(1, len(classes)+1))
    ax4.set_xticklabels(classes)
    ax4.set_xlabel('H-S order difference k', fontsize=36)
    ax4.set_yticks([45, 90])

    ax4.set_ylabel('Mean angle [°]', fontsize=36,labelpad=18)
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax4.tick_params(labelsize=size0, direction='in', width=1, length=8, pad=10)
    ax4.spines['bottom'].set_linewidth(boxwidth)
    ax4.spines['left'].set_linewidth(boxwidth)
    ax4.spines['right'].set_linewidth(boxwidth)
    ax4.spines['top'].set_linewidth(boxwidth)

    labels = ['a', 'c', 'b', 'd']
    positions = [ax1, ax2, ax3, ax4]  
    for label, ax in zip(labels, positions):
        ax.text(0.01, 0.97, label, transform=ax.transAxes,
                        fontsize=size0 + 8, fontweight='bold',
                        ha='left', va='top', color='black')
    plt.tight_layout()
    
    # ****************** figure 4 ********************************
    dfplot1 = classify_quantile(
        dfplot1, 
        'meanAI',
        labels=[
            rf'$\overline{{\mathrm{{AI}}}} ≤ {dfplot1["meanAI"].quantile(0.2):.2f}$',
            rf'{dfplot1["meanAI"].quantile(0.2):.2f} < $\overline{{\mathrm{{AI}}}} ≤ {dfplot1["meanAI"].quantile(0.5):.2f}$',
            rf'$\overline{{\mathrm{{AI}}}} > {dfplot1["meanAI"].quantile(0.5):.2f}$'
            ]
        )
    
    dfplot1 = classify_quantile(
        dfplot1, 
        'meanSlope',
        labels=[
            rf'$\overline{{\mathrm{{S}}}} ≤ {dfplot1["meanSlope"].quantile(0.2):.2f}$',
            rf'{dfplot1["meanSlope"].quantile(0.2):.2f} < $\overline{{\mathrm{{S}}}} ≤ {dfplot1["meanSlope"].quantile(0.5):.2f}$',
            rf'$\overline{{\mathrm{{S}}}} > {dfplot1["meanSlope"].quantile(0.5):.2f}$'
            ]
    )
    dfplot2 = classify_quantile(dfplot2, 'meanAI', labels=['Low', 'Medium', 'High'])
    dfplot2 = classify_quantile(dfplot2, 'meanSlope', labels=['Low', 'Medium', 'High'])

    q01, q99 = Tk['Tk_c'].quantile([0.01, 0.99])
    mid_bins = np.linspace(q01, q99, 9)
    bin_edges = np.concatenate(([-np.inf], [q01], mid_bins[1:-1], [q99], [np.inf]))
    Tk['Tk_bin'] = pd.cut(Tk['Tk_c'], bins=bin_edges, labels=False)

    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.1, wspace=0.05)
    rows = [0, 1]
    axes = [fig.add_subplot(gs[r, c]) for r in rows for c in range(2)]
    axes = np.array(axes).reshape(2, 2)
    size0 = 36
    palette1 = sns.color_palette("YlOrBr", n_colors=3)
    palette2 = sns.color_palette("BuGn", n_colors=3)

    # Side-branching vs AI
    custom_bin_and_plot(dfplot1, 'meanAI_class', (0.5, 4.7), (32, 80), axes[0][0], palette1)
    axes[0][0].set_ylabel('Mean\nside-branching\nangle [°]', fontsize=36)

    # Side-branching vs Slope
    custom_bin_and_plot(dfplot1, 'meanSlope_class', (0.5, 4.7), (32, 80), axes[0][1], palette2)

    # Bifurcation vs AI
    custom_bin_and_plot(dfplot2, 'meanAI_class', (0.5, 4.7), (32, 80), axes[1][0], palette1)
    axes[1][0].set_xlabel('Tokunaga parameter c', fontsize=36)
    axes[1][0].set_ylabel('Mean bifurcation\nangle [°]', fontsize=36)

    # Bifurcation vs Slope
    custom_bin_and_plot(dfplot2, 'meanSlope_class', (0.5, 4.7), (32, 80), axes[1][1], palette2)
    axes[1][1].set_xlabel('Tokunaga parameter c', fontsize=36)

    # ------ set legend ----------------
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg=fig.legend(
        handles, labels,
        fontsize=size0-5,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.25,
        loc='center',  
        bbox_to_anchor=(0.32, 0.5),  
        bbox_transform=fig.transFigure,  
    )
    frame = leg.get_frame()
    frame.set_edgecolor('black')   
    frame.set_linewidth(1.5)      
    frame.set_facecolor('white')  
    frame.set_alpha(1)             

    handles, labels = axes[0][1].get_legend_handles_labels()
    leg=fig.legend(
        handles, labels,
        fontsize=size0-5,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.25,
        loc='center',  
        bbox_to_anchor=(0.714, 0.5),  
        bbox_transform=fig.transFigure,
        frameon=True
    )

    frame = leg.get_frame()
    frame.set_edgecolor('black')   
    frame.set_linewidth(1.5)       
    frame.set_facecolor('white')   
    frame.set_alpha(1)             


    labels = ['a', 'b', 'c', 'd']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  

    for label, (i, j) in zip(labels, positions):
        axes[i][j].text(0.02, 0.97, label, transform=axes[i][j].transAxes,
                        fontsize=size0 + 8, fontweight='bold',
                        ha='left', va='top', color='black')

    axes[0][0].tick_params(axis='x', labelbottom=False)
    axes[0][1].tick_params(axis='x', labelbottom=False)
    axes[0][1].tick_params(axis='y', labelleft=False)
    axes[1][1].tick_params(axis='y', labelleft=False)
    axes[1][0].set_xticks([1,2,3,4])
    axes[1][1].set_xticks([1,2,3,4])

    plt.tight_layout()

    # ****************** figure 5 *******************************
    Tk_bins =[0.4,2.0,2.3,2.6,2.9,6.6]
    ecdfplot(Tk,'Tk_c', 5,  Tk_bins, 'Bifurcation')
    ecdfplot(Tk,'Tk_c', 5,  Tk_bins, 'Sidebranch')
    
    # ****************** statistics for figure 6 ****************
    df0 =Tk[['Tk_c','Sidebranch','AI','Slope','SR']]
    df0=df0.dropna()
    df0_ranked = df0.rank()
    partial_corr = pg.pcorr(df0_ranked).round(2)

    print(partial_corr)
        
    
if __name__ == '__main__':
    main()


