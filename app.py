from flask import Flask, render_template, request
from pickle import load
import numpy as np
import sklearn
from sklearn.preprocessing import FunctionTransformer
transformer = FunctionTransformer(np.log1p)
app = Flask(__name__)
model = load(open('le.pkl', 'rb'))
# load the scaler
scaler = load(open('ss.pkl', 'rb'))
@app.route('/')
def Home():
    return render_template('index.html') 


@app.route("/predict", methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        Type=request.form['Type']
        if(Type=='Apartment'):
            Type=0        
        else:
            Type=1
        Per_Sqft=float(request.form['Per_Sqft'])
        Per_Sqft=transformer.transform([Per_Sqft])
        Area = float(request.form['Area'])
        Area=scaler.transform(np.array([Area]).reshape(-1,1))
        Transaction=request.form['Transaction']
        if(Transaction=='New_Property'):
            Transaction=1        
        else:
            Transaction=0 
        Furnishing=request.form['Furnishing']
        if(Furnishing=='Unfurnished'):
            Furnishing=0        
        elif(Furnishing=='Semi-Furnished'):
            Furnishing=1        
        else:
            Furnishing=2
        Parking=request.form['Parking']   
        if(Parking=='1'):
            Parking=1
        elif(Parking=='2'):
            Parking=2
        elif(Parking=='3'):
            Parking=3
        else:
            Parking=4
                        
        prediction=model.predict([[Area,2,Furnishing,Parking,Transaction,Type,Per_Sqft]])
        output=round(prediction[0],2)
        return render_template('index.html',prediction_text=" Price of house  is {}".format(output))
    else:
        return render_template('index.html')    
if __name__=='__main__':
    app.run(debug=True)        
