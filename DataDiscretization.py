import pandas as pd 
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
from matplotlib import pyplot
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class Data_Discretization:
    """Transform data from numerical to level-based data.
    
    Firstly, the category data will be removed and concatenate at the end.
    
    Args:
        data (.csv): csv data file 
        method (string): k-mean, etc. 

    Returns:
        output: csv file with data discreted.
    """
     
    def __init__(self, dataset, target_col, drop_col, method='k-mean'):
        self.raw_dataset = pd.read_csv(dataset)
        self.target = pd.read_csv(dataset)[target_col]
        self.dataset = pd.read_csv(dataset).drop(drop_col, axis=1)
        self.header_dataset = pd.read_csv(dataset).drop(drop_col, axis=1).columns
        self.drop_col = drop_col
        self.method = method
        
    def discrete_data(self, export_csv=False):
        if self.method == 'k-mean':
            return self.k_mean(export_csv)
        elif self.method == 'fuzzy':
            return self.fuzzy(export_csv)
        
    def k_mean(self, export_csv):
        """A K-means discretization transform will attempt to fit k clusters for each input variable and then assign each observation to a cluster.
        """
        # perform a k-means discretization transform of the dataset
        trans = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
        data = trans.fit_transform(self.dataset)  # transformed [0, 1, 2]
        
        # convert the array back to a dataframe
        dataset = pd.DataFrame(data)
        
        # convert int to level
        numerical = {0.0: 'low', 1.0: 'medium', 2.0: 'high'}
        for attr in dataset:
            dataset[attr] = dataset[attr].map(numerical)
        dataset.columns = self.header_dataset
            
        other = {0.0: 'low', 1.0: 'high'}  # dict for work_accident and promotion_last_5years
        dataset['work_accident'] = self.raw_dataset['work_accident'].map(other)
        dataset['promotion_last_5years'] = self.raw_dataset['promotion_last_5years'].map(other)
        
        remain_col = ['left', 'sales', 'salary']
        for attr in remain_col:
            dataset[attr] = self.raw_dataset[attr]
        
        # export csv_file
        if export_csv:
            dataset.to_csv('DataDiscreted.csv', index=False)
            
        return dataset
        
    def fuzzy(self, export_csv):
        """Data transformation by applying Fuzzy set theory.

        Args:
            export_csv (bool): Do you want to export csv file?
        """
        fuzzy = {}
        feature_list = self.dataset.columns
        for feature in feature_list:
            key = feature
            
            # fuzzy and auto membership function population is possible with .automf (3, 5, or 7)
            element = ctrl.Antecedent(self.dataset[feature], key)
            element.automf(3)
            
            fuzzy = {key: element}
            print(element['average'])
        
        
# TODO: find the cutting points
# pd.cut(x=df['height'], bins=[0,25,50,100,200], labels=["very short", " short", "medium","tall"])