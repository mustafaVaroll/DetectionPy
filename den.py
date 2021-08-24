import pandas as pd
import os

filename='video.mp4'
os.system('python vehicle_detect.py ' +filename)

# csv_file="olcum.csv"
# data = pd.read_csv(os.path.join('upload', csv_file))

# data_columns = list(data.columns)
# data_size = data.size
# data_targetname = data[data.columns[-1]].name

# data_features = data_columns[0: -1]

# data_xfeatures = data.iloc[:, 0:-1]

# allmodels=[]




# for j in range(len(data)):
#         msg=""
#         for i in range(len(data_columns)):
#         #msg = "%s: %f: (%f)" % (name, cv_results.mean(), cv_results.std())
#           msg += str(data[data.columns[i]][j]) +" "
#         allmodels.append(msg) 
      
        
# print(allmodels)