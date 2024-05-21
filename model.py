from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
import numpy as np
app = Flask(__name__)
import pandas as pd
from joblib import dump
import pandas as pd
import numpy as py
import osmnx as ox
import networkx as nx
import folium
import random
from folium.plugins import HeatMap
import numpy as np
import requests
import json
import joblib
# Load the dataset (assuming df is already loaded)
# Replace 'your_dataset.csv' with the actual file path if needed
@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/criminal')
def criminal():
    
    return render_template('criminal.html')

@app.route('/victim')
def victim():
    
    return render_template('victim.html')
@app.route('/crimewatch')
def crimewatch():
    
    return render_template('crime.html')

@app.route('/demographics')
def demographics():
    
    return render_template('demographics.html')

@app.route('/crimetime')
def crimetime():
    
    return render_template('crime time.html')

@app.route('/crimetype')
def crimetype():
    
    return render_template('type.html')
@app.route('/beatpred')
def beatpred():
    
    return render_template('beat pred.html')

@app.route('/process_option', methods=['POST'])
def process_option():
    data = request.get_json()  # Parse the incoming JSON data
    number = data.get('num')
    datasets=['all_merged.csv','district_village_mapping.csv']
    selected_option = request.json['selectedOption']
    # Process the selected option here
    df = pd.read_csv(datasets[number])
   
    df1=df[df.District_Name == selected_option]
    df1['Village_Area_Name']=[x.upper() for x in  df1['Village_Area_Name']]
    df1=df1[~((df1['Village_Area_Name']=='OTHERS')|(df1['Village_Area_Name']=='OTHER'))]
    response={'items2':(df1['Village_Area_Name']).tolist()}
    return jsonify(response)

@app.route('/process_option1', methods=['POST'])
def process_option1():
    data = request.get_json()  # Parse the incoming JSON data
   
   
    selected_option = request.json['selectedOption']
    # Process the selected option here
    df = pd.read_csv('village_beat_mapping.csv')
 
    df1=df[df.Village_Area_Name == selected_option]
   
    response={'items2':(df1['Beat_Name']).tolist()}
    return jsonify(response)

@app.route('/page')
def page():
    return render_template('index.html')

    
@app.route('/predict1', methods=['POST'])
def predict1():
    

   
 
    # Take user input for the district name
    data = request.get_json()

    district_name=data["Area"]
    # Count the occurrences of each village name in the specified district
   
    outputs=['Caste_x','Profession_x','Sex_x','Caste_y','Profession_y','Sex_y']
    responses=[]
    print("hi")
    for out in outputs:
        nb_classifier = joblib.load ("demographics/"+out + "_model1" + district_name + ".joblib")
        print("hi")
        label_encoder = joblib.load ("demographics/"+out + "_encoder1" + district_name + ".joblib")
        sample_df = pd.DataFrame([data])
        print(data)  # Wrap data in a list to ensure DataFrame creation
    
        probabilities = nb_classifier.predict_proba(sample_df)
        print (probabilities )
        probabilities = probabilities[0] 
        if out=='Sex_x' or out=='Sex_y' :
            top_3_indices = np.argsort(probabilities)[::-1][:1]
            top_3_probabilities = probabilities[top_3_indices]
            top_3_castes = label_encoder.inverse_transform(top_3_indices)
        else:
            top_3_indices = np.argsort(probabilities)[::-1][:3]
            top_3_probabilities = probabilities[top_3_indices]
            top_3_castes = label_encoder.inverse_transform(top_3_indices)
        # Prepare response
        response = []
        for classes, probability in zip(top_3_castes, top_3_probabilities):
            response.append({'classes': classes.upper(), "Probability": probability})
        responses.append(response)
    print(responses)
    return jsonify(responses)

   

@app.route('/predict2', methods=['POST'])
def predict2():
    data = request.get_json()
    district_name=data["Area"]
    nb_classifier = joblib.load("crime time models/crime_time_model"+  district_name+".joblib")
    
    label_encoder = joblib.load("crime time models/crime_time_encoder"+  district_name+".joblib")
    sample_df = pd.DataFrame([data])
    

    print(data)  # Wrap data in a list to ensure DataFrame creation
    # Predict probabilities for each village area
    probabilities = nb_classifier.predict_proba(sample_df)
    
    probabilities = probabilities[0]  # Extract probabilities for the first sample
    # Get the top 3 predicted village areas
    top_3_indices = np.argsort(probabilities)[::-1][:10]
    top_3_probabilities = probabilities[top_3_indices]
    top_3_village_areas = label_encoder.inverse_transform(top_3_indices)
    # Prepare response
    response = []
    for village_area, probability in zip(top_3_village_areas, top_3_probabilities):
        response.append({"Village Area": village_area, "Probability": probability})
    return jsonify(response)


@app.route('/predict4', methods=['POST'])
def predict4():
    data = request.get_json()
    district_name=data["Area"]
    nb_classifier = joblib.load("crime type models/crime_type_model"+  district_name+".joblib")
    
    label_encoder = joblib.load("crime type models/crime_type_encoder"+  district_name+".joblib")
    sample_df = pd.DataFrame([data])
    

    print(data)  # Wrap data in a list to ensure DataFrame creation
    # Predict probabilities for each village area
    probabilities = nb_classifier.predict_proba(sample_df)
    
    probabilities = probabilities[0]  # Extract probabilities for the first sample
    # Get the top 3 predicted village areas
    top_3_indices = np.argsort(probabilities)[::-1][:10]
    top_3_probabilities = probabilities[top_3_indices]
    top_3_village_areas = label_encoder.inverse_transform(top_3_indices)
    # Prepare response
    response = []
    for village_area, probability in zip(top_3_village_areas, top_3_probabilities):
        response.append({"Village Area": village_area, "Probability": probability})
    return jsonify(response)

@app.route('/predict5', methods=['POST'])
def predict5():
   
    data = request.get_json()
    district_name=data["Area"]
    nb_classifier = joblib.load("beat_manage/beatmodel1"+district_name+".joblib")
    
    label_encoder = joblib.load("beat_manage/beatencoder1"+district_name+".joblib")
    sample_df = pd.DataFrame([data])
    nb_classifier1 = joblib.load("crime time models/crime_time_model"+  district_name+".joblib")
    
    label_encoder1 = joblib.load("crime time models/crime_time_encoder"+  district_name+".joblib")

    
    print(data)  # Wrap data in a list to ensure DataFrame creation
    # Predict probabilities for each village area
    probabilities = nb_classifier.predict_proba(sample_df)
    
    probabilities = probabilities[0]  # Extract probabilities for the first sample
    # Get the top 3 predicted village areas
    top_3_indices = np.argsort(probabilities)[::-1][:7]
    top_3_probabilities = probabilities[top_3_indices]
    top_3_village_areas = label_encoder.inverse_transform(top_3_indices)
    time=[]
   
    for x in top_3_village_areas:
        sample_df1=pd.concat([sample_df, pd.DataFrame({'Village_Area_Name':[x]})], axis=1)
        time.append(label_encoder1.inverse_transform(nb_classifier1.predict(sample_df1)))

    print(time)
    df=pd.read_csv("locations.csv")
    filtered_df=df[df['Village_Area_Name'].isin(top_3_village_areas)]
    filtered_df['Probability']=[top_3_probabilities[list(top_3_village_areas).index(x)] for x in filtered_df['Village_Area_Name']]
    

    
    def adjust_duplicates(lst):
        seen = {}
        adjusted_list = []
        
        for value in lst:
            if value in seen:
                # If the value is a duplicate, add a small random value between -0.5 and 0.5
                adjustment = random.uniform(-0.009, 0.009)
                new_value = value + adjustment
                # Ensure the new value is unique by re-checking the seen dictionary
                while new_value in seen:
                    adjustment = random.uniform(-0.009, 0.009)
                    new_value = value + adjustment
                seen[new_value] = True
                adjusted_list.append(new_value)
            else:
                # If the value is not a duplicate, add it to the adjusted list
                seen[value] = True
                adjusted_list.append(value)
        
        return adjusted_list
    
    filtered_df['Latitude']=adjust_duplicates(filtered_df['Latitude'])
    filtered_df['Longitude']=adjust_duplicates(filtered_df['Longitude'])
    
    m = folium.Map(location=[filtered_df['Latitude'].mean(), filtered_df['Longitude'].mean()], zoom_start=9,tiles='CartoDB dark_matter')

# Prepare data for the heatmap
    heat_data = [[row['Latitude'], row['Longitude'], row['Probability']] for index, row in filtered_df.iterrows()]
    gradient = {1: "#FF0000", 0.8: '#E32227', 0.6: 'crimson', 0.4: '#BF0000', 0.2: '#800000'}

# Add the heatmap layer to the map
    HeatMap(heat_data , gradient=gradient).add_to(m)
    for index, row in filtered_df.iterrows():

        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            tooltip=row['Village_Area_Name']
        ).add_to(m)
    map_html2 = m.get_root()._repr_html_()

    # Prepare response
    response = []
    print(time[0][0])
   
   


    for village_area, probability,t in zip(top_3_village_areas, top_3_probabilities,time):
        response.append({"Village Area": village_area, "Probability": probability,'Time':t[0]})
    df1=pd.read_csv("village_beat_mapping.csv")
    responses=[response,map_html2,list(top_3_village_areas),list(df1[df1['Village_Area_Name']==top_3_village_areas[0]]['Beat_Name'])]
    
    return jsonify(responses)

@app.route('/graph', methods=['POST'])
def graph():
    
    data = request.get_json()
    print(data)
    df=pd.read_csv("beats.csv")
    vil=data["Village_Area_Name"]
    beat=data["beat"]
    df=df[(df["Village_Area_Name"]==vil) & (df["Beat_Name"]==beat)]
    print(df)
  
    
    
    response=[list(df['Year']),list(df['count'])]


    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
    
