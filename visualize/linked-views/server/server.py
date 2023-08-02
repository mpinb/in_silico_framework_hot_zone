import os
import logging
import tempfile
import threading
import util
import glob
import json
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from werkzeug.serving import make_server


def load_tables(data_folder, config, max_num_rows = 50000):    
    csvFiles = glob.glob(os.path.join(data_folder,"*.csv"))
    tables = {}
    
    for filepath in csvFiles:   
        basename = os.path.basename(filepath) 

        print("start read: {}".format(basename))
        df = pd.read_csv(filepath, nrows=max_num_rows)    
        print("complete read")

        samplingSettings = config["sampling_settings"]
        if(basename in samplingSettings):
            settings = samplingSettings[basename]
            seed = settings["seed"]
            sampleSize = settings["number"]
            randomState = np.random.RandomState(seed)
            df = df.sample(sampleSize, random_state=randomState)

        columns_data = df.to_dict(orient="records")
        tables[basename] = {}
        tables[basename]["flat"] = df
        tables[basename]["records"] = columns_data

    return tables


def write_objects_to_file(filenameProjects, objects):
    with open(filenameProjects, 'w') as file:
        json.dump(objects, file)

def get_data_ranges(df):
    ranges = {}
    ranges["min"] = df.min().to_list()
    ranges["max"] = df.max().to_list()
    return ranges 

def normalize(df, low=-1, high=1):
    scaler = MinMaxScaler(feature_range=(low, high))
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized


def getPCA(df):    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    pca = PCA(2)
    pca.fit(scaled_data)
    principal_components = pca.transform(scaled_data)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_components = min_max_scaler.fit_transform(principal_components)

    return normalized_components.tolist()

    
class LinkedViewsServer:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)        
        self.thread = None        

        self.loadDataLegacy()

    def loadDataLegacy(self):
        self.resourceDir = self.data_folder/"resources"
        self.config = util.loadJson(self.data_folder/"config.json")
        self.tables = load_tables(self.data_folder, self.config)

        self.objects = []
        self.filenameProjects = self.data_folder/"projects.json"

        # Load objects from the file on server startup
        try:
            with open(self.filenameProjects, 'r') as file:
                self.objects = json.load(file)
        except FileNotFoundError:
            self.objects = []


    def start(self, port=5000):                        
        if(self.thread is not None):
            print(f"server already running at port {self.port}")
            return        
        
        try:
            self.app = Flask(__name__)       
            CORS(self.app)     
            self.server = make_server('127.0.0.1', port, self.app)                        
            self.port = port
            self.init_routes()
        except SystemExit as e:
            self.server = None
            return

        def _start():
            self.server.serve_forever()

        self.thread = threading.Thread(target=_start)      
        self.thread.start()
        print(f"server is running at port {self.port}")
        self.start_logging()
        print("set data:        server.set_data(df)")
        print("stop server:     server.stop()")
        

    def stop(self):
        if(self.thread is None):
            print("server is not running")
            return

        print(f"Stopping server. If operation does not terminate, try to repeatedly reopen http://127.0.0.1:{self.port} in the browser.")
        self.stop_logging()
        self.server.shutdown()                
        self.thread.join()
        self.thread = None
        print("Server stopped.")

    def start_logging(self):
        tmp_folder = Path(tempfile.gettempdir())
        log_filename = tmp_folder/"linked-views-server.log"
        print(f"logs are written to {log_filename}")
        self.logfile_handler = logging.handlers.RotatingFileHandler(
            filename=log_filename,
            mode='w',
            maxBytes=1048576
        )
        self.logfile_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
        self.logfile_handler.setFormatter(formatter)
        logging.getLogger().addHandler(self.logfile_handler)

    def stop_logging(self):
        logging.getLogger().removeHandler(self.logfile_handler)

    def set_data(self, df):
        assert self.server is not None
        raise NotImplementedError

    def init_routes(self):
        self.app.add_url_rule('/', 'index', self.index)        
        self.app.add_url_rule('/matrixServer/', 'get_objects', self.get_objects, methods=['GET'])
        self.app.add_url_rule('/matrixServer/<name>', 'delete_object', self.delete_object, methods=['DELETE'])
        self.app.add_url_rule('/matrixServer/', 'add_object', self.add_object, methods=['POST'])
        self.app.add_url_rule('/matrixServer/getMetaData', 'getMetaData', self.getMetaData, methods=["GET", "POST"])
        self.app.add_url_rule('/matrixServer/getResourceJSON', self.getResourceJSON, methods=["GET", "POST"])
        self.app.add_url_rule("/matrixServer/getValues", "getValues", self.getValues, methods=["GET", "POST"])

    def index(self):
        return f"Tables: {[name for name in self.tables.keys()]}"


    """
    ########################################################################################
                                    session storage
    ########################################################################################
    """

    def get_objects(self):
        return jsonify(self.objects)

    def delete_object(self, name):        
        self.objects = [obj for obj in self.objects if obj['name'] != name]
        write_objects_to_file(self.filenameProjects, self.objects)    
        return jsonify({'message': 'Object deleted'})
    
    def add_object(self):

        def delete_object_by_property(json_list, property_name, property_value):
            filtered_list = [obj for obj in json_list if obj.get(property_name) != property_value]
            return filtered_list
        
        new_object = request.get_json()

        self.objects = delete_object_by_property(self.objects, "name", new_object["name"])
        self.objects.append(new_object)
        write_objects_to_file(self.filenameProjects, self.objects)
        return jsonify(new_object)

    """
    ########################################################################################
                                    end of session storage
    ########################################################################################
    """


    #@app.route("/matrixServer/getMetaData", methods=["GET", "POST"])
    #@cross_origin()
    def getMetaData(self):        
        if request.method == "POST":
            if request.data:
                data = request.get_json(force=True)
                
                meta_data = []            
                for tableName, tableData in self.tables.items():
                    df = tableData["flat"]            
                    meta_data.append({
                        "name" : tableName,
                        "num_rows" : df.shape[0],
                        "columns" : df.columns.to_list(),
                        "data_ranges" : get_data_ranges(df)                
                    })

                response_data = {
                    "meta_data" : meta_data,
                    "available_views" : self.config["available_views"],
                    "cached_tables" : self.config["cached_tables"]
                }
                return json.dumps(response_data)


    #@app.route("/matrixServer/getResourceJSON", methods=["GET", "POST"])
    #@cross_origin()
    def getResourceJSON(self):        
        if request.method == "POST":
            if request.data:
                data = request.get_json(force=True)

                resourceName = data["filename"]
                filename = self.resourceDir/resourceName

                if(not os.path.exists(filename)):
                    raise ValueError(filename)
                else:
                    jsonData = util.loadJson(filename)

                    response_data = {
                        "filename" : resourceName,
                        "jsonData" : jsonData
                    }

                    return json.dumps(response_data)



    #@app.route("/matrixServer/getValues", methods=["GET", "POST"])
    #@cross_origin()
    def getValues(self):
        
        if request.method == "POST":
            if request.data:
                data = request.get_json(force=True)

                tableName = data["table"]
                df = self.tables[tableName]["flat"]

                columns = data["columns"]
                indices = data["indices"]
                format = data["format"]
                #print(columns)
                #print(set(columns) - set(df.columns))
                assert set(columns).issubset(set(df.columns))
                assert len(indices) == 0 or max(indices) < df.shape[0]            

                if(len(indices) == 0):
                    filtered_df = df[columns]
                else:
                    filtered_df = df.iloc[indices][columns]

                if(format == "expanded"):
                    values = filtered_df.to_dict(orient="records")
                elif(format == "flat"):
                    values = filtered_df.values.tolist()
                elif(format == "flat-normalized"):
                    values = normalize(filtered_df).values.tolist()
                elif(format == "flat-normalized-PCA"):
                    values = getPCA(filtered_df) 
                else:
                    raise ValueError(format)

                response_data = {
                    "columns" : columns,
                    "indices" : indices,
                    "values" : values,
                    "data_ranges" : get_data_ranges(filtered_df)
                }

                # -------
                """
                json_string = json.dumps(response_data)
                zip_buffer = BytesIO()    
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_STORED) as zip_file:
                    zip_file.writestr('data.json', json_string)

                response = make_response(zip_buffer.getvalue())
                response.headers['Content-Type'] = 'application/zip'
                response.headers['Content-Disposition'] = 'attachment; filename=data.zip'
                return response
                # -------
                """
                return json.dumps(response_data)
    