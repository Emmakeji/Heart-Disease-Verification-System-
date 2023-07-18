from django.shortcuts import render
import pickle as pkl
import numpy as np
import pandas as pd
# Create your views here.

def homepage(request):
    return render(request, 'index.html')

def predict_page(request):
    error = None
    if request.method == 'GET':
        return render(request, 'predict.html')
    elif request.method == 'POST':
        cp = request.POST['cp']
        thalach = request.POST['thalach']
        slope = request.POST['slope']
        RestECG = request.POST['RestECG']
        age = request.POST['age']
        fbs = request.POST['fbs']

        if int(fbs) > 120:
            fbs = 1
        else:
            fbs = 0

        data_entry = pd.DataFrame(
            {   
                'age': age,
                'sex': 1,
                'cp': cp,
                'trestbps':130,
                'chol': 240,
                'fbs': fbs,
                'restecg': RestECG,
                'thalach':thalach,
                'exang': 0,
                'oldpeak':0.8,
                'slope':slope,
                'ca': 0,
                'thal':2,	
            }, index = [1]
        )

        with open('model/standardScaler.pkl', 'rb') as file:
            scaler = pkl.load(file)
        df_scaled = scaler.transform(data_entry)
        df = pd.DataFrame(df_scaled, columns=scaler.feature_names_in_)

        features = ['cp', 'thalach', 'slope', 'restecg', 'age', 'fbs']
        df = df[features]

        print(df)

        with open('model/model.pkl', 'rb') as file:
            model = pkl.load(file)

        prediction = model.predict(df)
        
        context = {
            'prediction': prediction[0]
        }
        
        return render(request, 'result.html', context)

