
from flask import Flask, render_template, request, url_for
from flask_bootstrap import Bootstrap
from flask_uploads import UploadSet, configure_uploads, IMAGES, DATA, ALL
from flask_sqlalchemy import SQLAlchemy

from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage


import os
import datetime
import time


import pandas as pd
import numpy as np


from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)


files = UploadSet('files', ALL)
app.config['UPLOADED_FILES_DEST'] = 'upload'
configure_uploads(app, files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///upload/filestorage.db'

db = SQLAlchemy(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def datauploads():
    if request.method == 'POST' and 'video' in request.files:
        file = request.files['video']
        filename = secure_filename(file.filename)
        os.system('python vehicle_detect.py ' +filename)
        csv_file="olcum.csv"
        #file.save(os.path.join('upload', csv_file))
        fullfile = os.path.join('upload',csv_file)
        
     
        date = str(datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S"))
        
        data = pd.read_csv(os.path.join('upload', csv_file))
        data_shape = data.shape
        data_columns = list(data.columns)
        data_size = data.size
        data_targetname = data[data.columns[-1]].name
        data_features = data_columns[0: -1]
        data_xfeatures = data.iloc[:, 0:-1]
        data_ylabels = data[data.columns[-1]]
        
        data_table = data
        X = data_xfeatures
        y = data_ylabels
        allmodels=[]
        for j in range(len(data)):
            msg=""
            for i in range(len(data_columns)):
                msg +=str(data[data.columns[i]][j]) +" "
            
            allmodels.append(msg) 
     
    
        
        model_names=[]     
        
       

    return render_template('details.html', filename=filename,
                            date=date,
                            data_table=data,
                            data_shape=data_shape,
                            data_size=data_size,
                            data_columns=data_columns,
                            data_targetname=data_targetname,
                            model_results=allmodels,
                            model_names=model_names,
                            fullfile=fullfile)


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
