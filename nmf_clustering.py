import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import random
import seaborn
import geopandas as gp
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import matplotlib.cm
from matrix_operation import mat_opr
import kmeans_minus_minus
import importlib
importlib.reload(kmeans_minus_minus)
from kmeans_minus_minus import kmeans_minus_minus

#################################################

# A class to improve and automate the clustering
# methods I designed in Jupyter notebook

# Allows me to tweak parameters and automatically 
# get results 

# INPUT 
# data: Pandas dataframe or similar 
# (2d array, np array, etc. such that array[0] is the first column)
# (dictionary with {column name: [column values],...})
# rank: chosen rank
# clusters: chosen # of clusters
# cluster_method: (Optional) default 'kmeans++' OR 'kmeans--
# num_outliers: chosen number of outlier points if using kmeans--
# 
#################################################

class nmf_cluster(mat_opr):
    def __init__(self,data,rank,clusters, cluster_method, num_outliers, w_init = None, h_init = None):
        if isinstance(data, pd.DataFrame):
            mat_opr.__init__(self,data)
        elif isinstance(data, dict):
            mat_opr.__init__(self, pd.DataFrame.from_dict(data))
        else:
            mat_opr.__init__(self, pd.DataFrame(data).T)

        self.rank = rank
        self.clusters = clusters
        self.cluster_method = cluster_method
        self.num_outliers = num_outliers

        # Note that this is an initialization of W in the equation X = W * H
        # sckikit learn only allows customization of H so to customize W requires solving
        # X^T = H^T * W^T
        self.w_init = w_init
        self.h_init = h_init

        # Non negative matrix factorization:
        self.X = None
        self.Y = None
        if w_init is None:
            if h_init is None:
                self.X,self.Y = self.sci_nmf(self.rank, separate=True)
            else:
                self.X, self.Y = self.sci_nmf(self.rank, separate=True, h_init = h_init)
        else:
           # self.X,self.Y = self.sci_nmf(self.rank, separate=True, w_init = w_init, h_init = h_init)
            self.X,self.Y = self.sci_nmf(self.rank, separate=True, w_init = True, h_init = np.transpose(w_init))
            

        dotted = pd.DataFrame(np.dot(self.X,self.Y))
        dotted.index = self.dataframe.index
        dotted.columns = self.dataframe.columns
        self.nmf = mat_opr(dotted)

        # pass Y from decomp into a dataframe object (easier to work with)
        indexer = []
        for i in range(self.rank):
            indexer.append("basis " + str(i))
        self.y_table = pd.DataFrame(self.Y, index=indexer)
        




    def cluster(self,num_clust=None):
        # Function to perform clustering
        # The method used is based on the value passed to self.cluster_method

        # Perform a clustering on the columns of Y after decomposition:
        if num_clust is None:
            num_clust = self.clusters

        if self.cluster_method is None or self.cluster_method == 'kmeans':
            # Normal scikit learn version of kmeans++
            y_clust = np.transpose(np.array(self.Y))
            kmeans = KMeans(n_clusters=num_clust, max_iter = 1000, tol = 1e-9).fit(y_clust)
            n_counter = pd.Series(kmeans.labels_)
            self.outliers = []

        elif self.cluster_method == 'kmeans--':
            # My own implementation of kmeans--
            arr = self.dataframe.T.values.tolist()
            minmin = kmeans_minus_minus(arr, num_clust, self.num_outliers)
            clust_labels = minmin.cluster()
            n_counter = pd.Series(clust_labels)
            
            self.outliers = []
            for o in np.where(clust_labels == -1)[0]:
                colo = self.dataframe.columns[o]
                self.outliers.append(colo)
                #n_counter = n_counter.drop(o,axis=0)

        else:
            raise ValueError("cluster has no method '" + str(self.cluster_method) + "'")


        return n_counter





    def sort_by_cluster(self, labels):
        #helper function to see which columns are in which cluster
        # takes as input a list of labels such that labels[0] = the cluster of column 0
        # Ouputs a dict of form {cluster # : [columns belonging to the cluster]}

        cluster_dict = {lab:[] for lab in labels.value_counts().index}
        for i in labels.index:
            cluster_dict[labels[i]].append(i) 

        return cluster_dict




    def plot_cluster(self, data_obj, labels, mean = False, sample_size = 10,legend=False, 
                 ylimit=None, axer=None, ylabel=None, title = None):
    
        # plots a sample of data (or the mean of that sample) based on clustering results

        subs = labels.value_counts()
        if axer is None:
            fig, axer = plt.subplots(len(subs), figsize=(14,12))
            
        if ylimit is not None:
            for a in axer:
                a.set_ylim(ylimit)     
                
        clust_dict = self.sort_by_cluster(labels)

        for i in clust_dict.keys():
            if sample_size > len(clust_dict[i]):
                for j in clust_dict[i]:
                    data_obj.iloc[:,j].plot(ax=axer[i], legend = legend)
            else:
                samples = random.sample(clust_dict[i], sample_size)
                for j in samples:
                    data_obj.iloc[:,j].plot(ax=axer[i], legend=legend)
            if title is not None:
                axer[i].set_title("Cluster " + str(i) + " " + title)
            else:
                axer[i].set_title("Cluster " + str(i))
                
            if ylabel is not None:
                axer[i].set_ylabel(ylabel)
                
        if mean:
            for k in clust_dict.keys():
                meaner = data_obj.iloc[:,clust_dict[k]].mean(axis=1)
                meaner.plot(ax=axer[k], color='k', linewidth=4.0, label='center', legend=legend)




    def pairplotter(self, num_tries):
        # takes a pandas dframe where one column contains the info about cluster labels
        # + number of different clusters to try

        # Outputs a seaborn pairplot of the data where the clusters are colored accordingly

        for n in num_tries:
            n_counter = self.cluster(num_clust = n)
            n_drop = n_counter
            for o in n_counter.loc[n_counter == -1].index:
                n_drop = n_drop.drop(o,axis=0)

            to_drop = []
            for o in self.outliers:
                to_drop.append(self.dataframe.columns.get_loc(o))

            n_cluster_df = self.y_table.T
            n_cluster_df['cluster'] = n_counter 
            n_drop_df = self.y_table.drop(to_drop, axis=1).T

            silhouette_avg = silhouette_score(np.array(n_drop_df.values.tolist()), n_drop)
            print("For n_clusters =", n, "The average silhouette_score is :", silhouette_avg)

            pp = seaborn.pairplot(n_cluster_df, hue="cluster", diag_kind="hist", height=2, aspect=1, palette='tab20')
            fig = pp.fig
            fig.subplots_adjust(top=0.93, wspace=0.3)
            t = fig.suptitle(str(n) + " Clusters", fontsize=16)
            plt.show()

            # print outliers that were removed if any
            if self.num_outliers is not None:
                print('Outliers: ')
                print(self.outliers)


    def pairwise_plotter(self, num_tries):
        # takes a number of clusters to try ex) pairwise plotter (6) outputs clustering for 2 - 6 clusters
        # ouptuts a heatmap of the pairwise distance matrix for the clustered input + clustered factor Y
        # Note: this plot does not include any outliers

        # this one is designed to be used separately from the rest
        # It takes a while because I did the simple O(n^2) implementation

        # rearrange pair_dist so that all columns/rows belonging to cluster 0 come before the ones correspoinding to cluster 1, etc.
        # graph and expect to see small distance on the diagonals and larger everywhere else
        for r in num_tries:
            fig, axes = plt.subplots(1, 2,figsize=(10,5))

            labels = self.cluster(num_clust = r)
            for o in labels.loc[labels == -1].index:
                labels = labels.drop(o,axis=0)

            input_pair_dist = self.pairwise_distance()
            input_pair_dist.columns = range(len(input_pair_dist.columns))
            factor_pair_dist =  mat_opr(self.y_table).pairwise_distance()

            order = []
            for h in range(r):
                order += labels.loc[labels == h].index.to_list()

            input_pair_dist = input_pair_dist[order]
            input_pair_dist = input_pair_dist.reindex(order)
            factor_pair_dist = factor_pair_dist[order]
            factor_pair_dist = factor_pair_dist.reindex(order)

            ip = seaborn.heatmap(input_pair_dist, ax = axes[0], xticklabels= False, yticklabels = False)
            fp = seaborn.heatmap(factor_pair_dist, ax = axes[1], xticklabels= False, yticklabels = False)

            # lot of work to get the x/yticks nice
            tot = len(input_pair_dist.columns)
            fracs = [va/tot for va in labels.value_counts().to_list()]
            #l_m = round(np.lcm.reduce(counti.value_counts().to_list())/tot)

            for axy in axes:
                axy.xaxis.set_major_locator(plt.MaxNLocator(10))
                axy.yaxis.set_major_locator(plt.MaxNLocator(10))

            so_far = 0
            labs = ['' for i in range(10)]
            for f in range(len(fracs)):
                pos = round(10*(fracs[f]/2 + so_far))
                labs[pos - 1] = 'Cluster ' + str(f)
                so_far += fracs[f]


            axes[0].set_xticklabels(labs, rotation= 45);
            axes[1].set_xticklabels(labs, rotation= 45);
            axes[0].set_yticklabels(labs);
            axes[1].set_yticklabels(labs);

            axes[0].set_title('Pairwise distance on Input')
            axes[1].set_title('Pairwise distance on Y')

            fig.suptitle(str(r) + " Clusters")

            # print outliers that were removed if any
            if self.num_outliers is not None:
                print('Outliers: ')
                print(self.outliers)


    def basis_vectors(self, comparer = None, label = ''):
        # plot basis vectors of X
        if self.rank % 2 == 1:
            square = (self.rank + 1)//2
        else:
            square = self.rank//2

        fig, axys = plt.subplots(square, 2, figsize=(12,9),constrained_layout=True)

        baser = pd.DataFrame(self.X)
        baser.index = pd.to_datetime(self.dataframe.index)

        tp = baser.max().max()
        bt = baser.min().min()
        for b in baser.columns:
            if comparer is not None:
                minner = 0
                dister = -1
                for c in comparer.columns:
                    dist = np.linalg.norm(baser[b] - comparer[c])
                    if dist < dister or dister == -1:
                        dister = dist
                        minner = c
                comparer[minner].plot(ax=axys.flatten()[b])
                
            baser[b].plot(ax=axys.flatten()[b], title = "Basis " + str(b))

            axys.flatten()[b].set_ylim([bt - (bt * 0.2) - 0.02, tp + (tp * 0.2)])
            axys.flatten()[b].legend([label, 'County'])



    def basis_cluster(self, labels, fig, grid):
        # plots both the basis vectors of X along with
        # the clustering of Y
        nrows_clust = int(grid.nrows/self.clusters)
        nrows_rank = int(grid.nrows/self.rank)

        # plot clustering of Y
        taken = 0
        axos = []
        #Make the subplots
        while taken != (nrows_clust * self.clusters):
            bx = fig.add_subplot(grid[taken:(taken + nrows_clust), 0])
            axos.append(bx)
            taken += nrows_clust

        top = self.y_table.max().max()
        bot = self.y_table.min().min()
        self.plot_cluster(self.y_table, labels, mean=True, 
            ylimit=[bot - (bot * 0.1), top + (top * 0.1)], axer=axos)

        # plot basis vectors of X
        taker = 0
        axys = []
        while taker != (nrows_rank * self.rank):
            cx = fig.add_subplot(grid[taker:(taker + nrows_rank), 1])
            axys.append(cx)
            taker += nrows_rank

        baser = pd.DataFrame(self.X)
        baser.index = self.dataframe.index
        tp = baser.max().max()
        bt = baser.min().min()
        for b in baser.columns:
            baser[b].plot(ax=axys[b], title = "Basis " + str(b))
            axys[b].set_ylim([bt - (bt * 0.2) - 0.02, tp + (tp * 0.2)])



    def case_cluster(self, labels, fig2, grid, nrows_clust=1, start=0):
        # plot the clustering results with respect to the original cumulative case data
        # as well as the calculated new cases/day data
 
        axer1 = []
        axer2 = []
        taken = start
        while taken < ((nrows_clust * self.clusters) + start):
            bx = fig2.add_subplot(grid[taken:(taken + nrows_clust),0])
            cx = fig2.add_subplot(grid[taken:(taken + nrows_clust),1])
            axer1.append(bx)
            axer2.append(cx)
            taken += nrows_clust

        top = self.dataframe.max().max()
        bot = self.dataframe.min().min()
        self.plot_cluster(self.dataframe, labels, mean=True, axer=axer1, ylabel="Normalized Cases", legend=False,
            title = "-- Cumulative Cases", ylimit = [bot - (bot * 0.1), top + (top * 0.1)])

        new_case_frame = self.new_case_calc()
        tp = new_case_frame.dataframe.max().max()
        bt = new_case_frame.dataframe.min().min()
        self.plot_cluster(new_case_frame.dataframe, labels, mean=True, legend=False,
            axer= axer2, ylabel="Normaized New Cases", title = "-- New Cases/Day",
            ylimit = [bt - (bt * 0.2), tp + (tp * 0.2)])




    def state_map(self, labels, fig, grid, start=0, spacing=1):
        # a function to make a plot of the US states colored by cluster
        
        cluster_by_state = {}
        for c in labels.index:
            s_name = self.dataframe.iloc[:,c].name
            cluster_by_state[s_name] = labels[c]

        # json file with geographic info for each state -- required for geopandas
        state_map = gp.read_file("US_States_geojson.json")

        cluster_col = []
        for i in state_map["NAME"]:
            try:
                cluster_col.append("Cluster " + str(cluster_by_state[i]))
            except:
                if i in self.outliers:
                    cluster_col.append("Outlier")
                else:
                    cluster_col.append(np.nan)

        state_map['cluster'] = cluster_col

        sx1 = fig.add_subplot(grid[start:(start + spacing * 3), 0:2]) #for most states

        sx2 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2), 0]) # alaska
        sx3 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2):, 1]) # Hawaii
        sx2.set_xlim(-200,-100)
        sx3.set_xlim(-165,-150)
        sx3.set_ylim(18,24)

        # specific color map "summer"
        cmapper = matplotlib.cm.get_cmap('summer')
        add_out = 0
        if self.num_outliers is not None:
            add_out = 1
        cspace = np.linspace(0,0.99, self.clusters + add_out)

        #color_dict = {i:ListedColormap(cmapper(cspace[i])) for i in range(self.clusters + add_out)}
        valz = list(state_map['cluster'].unique())
        try:
            valz.remove(np.nan)
        except:
            pass
        valz.sort()
        color_dict = {valz[i]:ListedColormap(cmapper(cspace[i])) for i in range(len(valz))}

        # plot using geopandas .plot()
        state_map[state_map['NAME'].isin(['Alaska','Hawaii']) == False].plot(column='cluster',
            ax=sx1, legend=True, categorical=True, figsize=(60,60), cmap='summer')

        if 'Alaska' in self.dataframe.columns:
            state_map[state_map['NAME'] == 'Alaska'].plot(column='cluster', ax=sx2, legend=True, 
                categorical=True, figsize=(30,30), cmap=color_dict[state_map.loc[state_map.NAME == 'Alaska', 'cluster'].to_list()[0]])

        if 'Hawaii' in self.dataframe.columns:
            state_map[state_map['NAME']=='Hawaii'].plot(column='cluster', ax=sx3, legend=True, 
                categorical=True, figsize=(30,30), cmap=color_dict[state_map.loc[state_map.NAME == 'Hawaii', 'cluster'].to_list()[0]])

        if 'Puerto Rico' in self.dataframe.columns:
            sx4 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2):, 2]) # Puerto Rico
            sx4.set_xlim(-68,-64)
            state_map[state_map['NAME']=='Puerto Rico'].plot(column='cluster', ax=sx4, legend=True, categorical=True, figsize=(30,30), 
            cmap=color_dict[state_map.loc[state_map.NAME == 'Puerto Rico', 'cluster'].to_list()[0]])


    




    def state_map_basis(self):
        # a function to make a plot of the US states colored by their factorization
        g_max = self.y_table.max().max() 
        g_min = self.y_table.min().min() 

        for rows in self.y_table.index:
            labels = self.y_table.loc[rows, :]
            fig = plt.figure(constrained_layout=True, figsize=(20,10))
            fig.suptitle(rows, size='x-large')
            grid = fig.add_gridspec(ncols = 3, nrows = 5)
            start = 0
            spacing = 1

            
            cluster_by_state = {}
            for c in labels.index:
                s_name = self.dataframe.iloc[:,c].name
                cluster_by_state[s_name] = labels[c]

            # json file with geographic info for each state -- required for geopandas
            state_map = gp.read_file("US_States_geojson.json")

            cluster_col = []
            for i in state_map["NAME"]:
                try:
                    cluster_col.append(cluster_by_state[i])
                except:
                    cluster_col.append(np.nan)

            state_map['cluster'] = cluster_col

            sx1 = fig.add_subplot(grid[start:(start + spacing * 3), 0:2]) #for most states

            sx2 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2), 0]) # alaska
            sx3 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2):, 1]) # Hawaii
            sx2.set_xlim(-200,-100)
            sx3.set_xlim(-165,-150)
            sx3.set_ylim(18,24)

            # specific color map "Blues"
            cmapper = matplotlib.cm.get_cmap('inferno')
            cspace = np.linspace(0,0.99, 100)

            #maxer = labels.max().max()
            #minner = labels.min().min()


            # plot using geopandas .plot()
            state_map[state_map['NAME'].isin(['Alaska','Hawaii', 'Puerto Rico']) == False].plot(column='cluster',
                ax=sx1, legend=True, vmax = g_max,vmin = g_min, figsize=(60,60), cmap='inferno')
            
            
            if 'Alaska' in self.dataframe.columns:
                al_val = state_map.loc[state_map['NAME'] == 'Alaska']['cluster']
                state_map[state_map['NAME'] == 'Alaska'].plot(column='cluster', ax=sx2, legend=False, figsize=(30,30), cmap = 'inferno', vmax = g_max, vmin = g_min)
                #cmap=ListedColormap(cmapper(cspace[int((al_val - g_min)/(g_max - g_min) * 100) - 1])))

            if 'Hawaii' in self.dataframe.columns:
                h_val = state_map.loc[state_map['NAME'] == 'Hawaii']['cluster']
                state_map[state_map['NAME']=='Hawaii'].plot(column='cluster', ax=sx3, legend=False, figsize=(30,30), cmap = 'inferno', vmax = g_max, vmin = g_min)
                #cmap=ListedColormap(cmapper(cspace[int((h_val - g_min)/(g_max - g_min) * 100) - 1])))

            if 'Puerto Rico' in self.dataframe.columns:
                p_val = state_map.loc[state_map['NAME'] == 'Puerto Rico']['cluster']
                sx4 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2):, 2]) # Puerto Rico
                sx4.set_xlim(-68,-64)
                state_map[state_map['NAME']=='Puerto Rico'].plot(column='cluster', ax=sx4, legend=False, figsize=(30,30), cmap = 'inferno', vmax = g_max, vmin = g_min)
                #cmap=ListedColormap(cmapper(cspace[int((p_val - g_min)/(g_max - g_min) * 100) - 1])))


    def state_per_capita(self):
        # similar to the above functions but just for graphing US map once
        # This is bad coding but I'm going to leave it for now.

        us_norm = self.dataframe

        cluster_by_state = {}
        for c in us_norm.columns:
            cluster_by_state[c] = us_norm.loc[:,c].sum()

        g_max = max(cluster_by_state.values())
        g_min = min(cluster_by_state.values())

        # json file with geographic info for each state -- required for geopandas
        state_map = gp.read_file("US_States_geojson.json")

        cluster_col = []
        for i in state_map["NAME"]:
            try:
                cluster_col.append(cluster_by_state[i])
            except:
                cluster_col.append(np.nan)

        state_map['cluster'] = cluster_col

        fig = plt.figure(constrained_layout=True, figsize=(20,10))
        fig.suptitle("US Cumulative Cases per capita", size='x-large')
        grid = fig.add_gridspec(ncols = 3, nrows = 5)
        start = 0
        spacing = 1

        sx1 = fig.add_subplot(grid[start:(start + spacing * 3), 0:2]) #for most states

        sx2 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2), 0]) # alaska
        sx3 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2):, 1]) # Hawaii
        sx2.set_xlim(-200,-100)
        sx3.set_xlim(-165,-150)
        sx3.set_ylim(18,24)

        # specific color map "Inferno"
        cmapper = matplotlib.cm.get_cmap('inferno')
        cspace = np.linspace(0,0.99, 100)

        # plot using geopandas .plot()
        state_map[state_map['NAME'].isin(['Alaska','Hawaii', 'Puerto Rico']) == False].plot(column='cluster',
            ax=sx1, legend=True, vmax = g_max,vmin = g_min, figsize=(60,60), cmap='inferno')
        
        
        if 'Alaska' in self.dataframe.columns:
            al_val = state_map.loc[state_map['NAME'] == 'Alaska']['cluster']
            state_map[state_map['NAME'] == 'Alaska'].plot(column='cluster', ax=sx2, legend=False, figsize=(30,30), cmap = 'inferno', vmax = g_max, vmin = g_min)
            #cmap=ListedColormap(cmapper(cspace[int((al_val - g_min)/(g_max - g_min) * 100) - 1])))

        if 'Hawaii' in self.dataframe.columns:
            h_val = state_map.loc[state_map['NAME'] == 'Hawaii']['cluster']
            state_map[state_map['NAME']=='Hawaii'].plot(column='cluster', ax=sx3, legend=False, figsize=(30,30), cmap='inferno',vmax = g_max, vmin = g_min)
            #ListedColormap(cmapper(cspace[int((h_val - g_min)/(g_max - g_min) * 100)])))

        if 'Puerto Rico' in self.dataframe.columns:
            p_val = state_map.loc[state_map['NAME'] == 'Puerto Rico']['cluster']
            sx4 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2):, 2]) # Puerto Rico
            sx4.set_xlim(-68,-64)
            state_map[state_map['NAME']=='Puerto Rico'].plot(column='cluster', ax=sx4, legend=False, figsize=(30,30), cmap = 'inferno', vmax = g_max, vmin = g_min)








# SAme thing but for US Counties
    def county_map_basis(self):
        # a function to make a plot of the US counties colored by their NMF factorization
        vmax = self.y_table.max().max() 
        vmin = self.y_table.min().min() 
        # json file with geographic info for each state -- required for geopandas
        state_map = gp.read_file("US_States_geojson.json")
        county_map = gp.read_file("US-counties.geojson")

        for rows in self.y_table.index:
            labels = self.y_table.loc[rows, :]
            fig = plt.figure(constrained_layout=True, figsize=(20,10))
            fig.suptitle(rows, size='x-large')
            grid = fig.add_gridspec(ncols = 3, nrows = 5)
            start = 0
            spacing = 1

            
            cluster_by_state = {}
            for c in labels.index:
                s_name = self.dataframe.iloc[:,c].name
                cluster_by_state[s_name] = labels[c]


            cluster_col = []
            for i in county_map.index:
                county = county_map.iloc[i,:]
                county_name = county["NAME"]
                state_num = county["STATE"]
                state_name = state_map.loc[state_map.STATE == state_num].NAME
                state_name = state_name.loc[state_name.index[0]]
                fips = int(county["id"])

                try:
                    cluster_col.append(cluster_by_state[(state_name, county_name,fips)])
                except:
                    cluster_col.append(np.nan)

            county_map['cluster'] = cluster_col

            sx1 = fig.add_subplot(grid[start:(start + spacing * 3), 0:2]) #for most states

            sx2 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2), 0]) # alaska
            sx3 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2):, 1]) # Hawaii
            sx2.set_xlim(-200,-100)
            sx3.set_xlim(-165,-150)
            sx3.set_ylim(18,24)


            # plot using geopandas .plot()
            county_map[~county_map['STATE'].isin(['02','15','72'])].plot(column='cluster',
                ax=sx1, legend=True, figsize=(60,60), cmap='inferno', vmax = vmax, vmin=vmin)

            
            states = self.dataframe.columns.get_level_values("state").value_counts().index
            #county_map[['cluster']] = county_map[['cluster']].fillna(0)
            
            if 'Alaska' in states:
                county_map.loc[county_map['STATE'] == '02'].dropna().plot(column='cluster', ax=sx2, legend=False, figsize=(30,30), cmap='inferno', vmax = vmax, vmin=vmin) 

            if 'Hawaii' in states:
                county_map.loc[county_map['STATE']=='15'].dropna().plot(column='cluster', ax=sx3, legend=False, figsize=(30,30), cmap='inferno', vmax = vmax, vmin=vmin)
            
            
            if 'Puerto Rico' in states:
                sx4 = fig.add_subplot(grid[(start + spacing * 3):(start + spacing * 3 + spacing * 2):, 2]) # Puerto Rico
                sx4.set_xlim(-68,-64)
                county_map[county_map['STATE']=='72'].plot(column='cluster', ax=sx4, legend=False, figsize=(30,30), cmap = 'inferno', vmax = vmax, vmin=vmin)
            
            




        
    def state_results(self):

        # cluster
        cluster_labels = self.cluster()
        non_outliers = cluster_labels
        for o in cluster_labels.loc[cluster_labels == -1].index:
                non_outliers = non_outliers.drop(o,axis=0)

        height = 9
        if self.clusters >= 5:
            height = 20

        # first cluster plot
        fig1 = plt.figure(constrained_layout=True, figsize=(12,height))
        nrows = np.lcm(self.clusters, self.rank)
        grid1 = fig1.add_gridspec(ncols=2, nrows=nrows)
        self.basis_cluster(non_outliers, fig1, grid1)
        fig1.suptitle("Initial Cluster on the Basis Vectors of Y", size='x-large')
        plt.show()
        print()

        # second cluster plot
        fig2 = plt.figure(constrained_layout=True, figsize = (12,height))
        grid2 = fig2.add_gridspec(ncols = 2, nrows = self.clusters)
        self.case_cluster(non_outliers, fig2, grid2)
        fig2.suptitle("Results of basis clustering applied to the original data", size='x-large')
        plt.show()
        print()

        # map plot
        fig3 = plt.figure(constrained_layout=True, figsize=(20,10))
        grid3 = fig3.add_gridspec(ncols = 3, nrows = 5)
        self.state_map(cluster_labels, fig3, grid3)
        fig3.suptitle("Geographical correlation of clustering", size='x-large')
        plt.show()






