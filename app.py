import numpy as np
from flask import Flask, request, render_template
import pickle
#Create an app object using the Flask class. 
app = Flask(__name__)
#Load the trained model. (Pickle file)
model = pickle.load(open('model.pkl', 'rb'))
import iris_model
@app.route('/', methods=['GET', 'POST'])
def basic():
    if request.method == 'POST':
        sepal_length = request.form['sepallength']
        sepal_width = request.form['petalwidth']
        petal_length = request.form['petallength']
        petal_width = request.form['petalwidth']
        y_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
        trained_model = iris_model.training_model()
        prediction_value = trained_model.predict(y_pred)
        setosa = 'Setosa'
        versicolor = 'versicolor'
        virginica ='virginica'
        if prediction_value == 0:
            return render_template('index.html', setosa=setosa)
        elif prediction_value == 1:
            return render_template('index.html', versicolor=versicolor)
        else:
            return render_template('index.html', virginica=virginica) 
    # return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)