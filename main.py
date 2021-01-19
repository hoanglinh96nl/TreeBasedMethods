from CalEntropy import Information_Gain

raw_data = 'recursos_humanos.csv'

entropy = Information_Gain(dataset=raw_data, target_col='left', drop_col=['no', 'left'])
entropy.entropy_info_gain()
