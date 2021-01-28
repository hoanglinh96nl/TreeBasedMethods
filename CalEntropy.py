import csv
import json
import numpy as np
import math
import pandas as pd 
import Utilities as util

class Information_Gain:
    """Calculate entropy for splitting criteria as information gain."""
    
    def __init__(self, dataset, target_col, drop_col):
        """Calculate entropy for splitting criteria as information gain.

        Args:
            dataset (.csv): dataset 
            target_col (list): name of target column
            drop_col (list): which columns should be deleted
        """
        self.target = pd.read_csv(dataset)[target_col]
        self.dataset = pd.read_csv(dataset).drop(drop_col, axis=1)

    def cal_entropy(self, column, target):
        """Calculate information gain for each column. In this problem, there are 2 value for target column (1, 0).

        Args:
            column (string): name of the attribute. ['name']
            target (list): contain name of target column and corresponding category values. ['name', [0, 1]]

        Returns:
            entropy: value of information gain of such attribute.
        """
        # We will need to find the percentage of each case in the column. numpy.bincount() return value is a NumPy array which will store the count of each unique value from the column that was passed as an argument.
        # counts = np.bincount(self.dataset[column])
        unique = self.dataset[column].unique().tolist()  # array of unique values in specific columns
        # probabilities of each unique value. Output = [0.1, 0.3]
        total_instance = len(self.dataset[column])

        # calculate entropy of each category of the column
        entropy = 0

        # calculate Info(A) for each unique value
        info_A = [0 for i in range(len(unique))]
        
        for uq in range(len(unique)):  # for each category in attribute column
            count_0, count_1, count_unique = 0, 0, 0
            
            for instance in range(len(self.dataset[column])):
                if self.dataset[column][instance] == unique[uq] and self.target[instance] == target[1][0]:
                    count_0 += 1
                else:
                    count_1 += 1
                count_unique += 1
            print(count_0)
            info_A[uq] = -(count_0/count_unique)*math.log(count_0/count_unique, 2) \
                - (count_1/count_unique)*math.log(count_1/count_unique, 2)

            entropy += (count_unique/total_instance)*info_A[uq]
        
        return entropy

    def entropy_info_gain(self):
        """A Python Function for Entropy."""
        # calculate total number of instance for specific attribute
        total_instance = len(self.dataset)

        # calculate Info(D) -> Evaluate the divergence
        # If the probability of each category is equal, the entropy value is 1, indicating that the classification information is the most cluttered.
        info_D = 0
        count_target = util.count_unique_value(self.target)  # count of each unique value 
        print(count_target)
        for target in count_target:
            info_D -= (target/total_instance)*math.log(target/total_instance, 2)

        entropy = []
        for attr in self.dataset.columns:  
            entropy.append(self.cal_entropy(attr, target=['left', [0, 1]]))  # attr: str
        

    def convert_to_json(self, index, save_file=True):
        """Convert csv file from dataset to json file.
        
        - index: primary key for convert from csv -> json
        - save_file: do you want to save json file? (True/False)
        """
        # create a dictionary 
        data = {} 
        
        # Open a csv reader called DictReader 
        with open(self.dataset, encoding='utf-8') as csvf: 
            csvReader = csv.DictReader(csvf) 
            
            # Convert each row into a dictionary and add it to data 
            for rows in csvReader: 
                
                # Assuming a column named 'No' to be the primary key 
                key = rows[index] 
                data[key] = rows 

        if save_file:
            with open('dataset.json', 'w+', encoding='utf-8') as outfile:
                outfile.write(json.dumps(data, indent=4)) 

        return data
