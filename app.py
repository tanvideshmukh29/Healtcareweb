from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd


import re
import math
app = Flask(__name__)






@app.route('/')
def home():
    return render_template("main.html")



@app.route('/breastcancer')
def breastcancer():
    return render_template("cancer.html")



@app.route('/cancerp',methods=['POST','GET'])
def cancerp():

        model2 = pickle.load(open('modelbc1.pkl', 'rb'))

        features2 = [int(x) for x in request.form.values()]

        final2 = [np.array(features2)]

        print(features2)

        print(final2)

        prediction2 = model2.predict(final2)

        output = prediction2

        if output == 1:

            return render_template('rheart.html', pred='The patient is diagnosed with heartpeoblem. ')

        else:
            return render_template('rheart.html', pred='The patient is not diagnosed with heartproblem.')



@app.route('/diabetes')
def diabetes():
    return render_template("diabetes.html")

@app.route('/diabete',methods=['POST','GET'])
def diabete():
    model2= pickle.load(open('modeldiabetes.pkl', 'rb'))

    features2=[int(x) for x in request.form.values()]

    final2=[np.array(features2)]

    print(features2)

    print(final2)

    prediction2=model2.predict(final2)


    output =  prediction2

    if output==1:

        return render_template('rdiabetes.html', pred='The patient is diagnosed with Diabetes. ')

    else:
        return render_template('rdiabetes.html',pred='The patient is not diagnosed with Diabetes .')


@app.route('/heart')
def heart():
    return render_template("heart.html")

@app.route('/heartpredict',methods=['POST','GET'])
def heartpredict():
    model2= pickle.load(open('modelheart.pkl', 'rb'))

    features2=[int(x) for x in request.form.values()]

    final2=[np.array(features2)]

    print(features2)

    print(final2)

    prediction2=model2.predict(final2)


    output =  prediction2

    if output==1:

        return render_template('rheart.html', pred='The patient is diagnosed with heartpeoblem. ')

    else:
        return render_template('rheart.html',pred='The patient is not diagnosed with heartproblem.')


@app.route('/kidney')
def kidney():
    return render_template("kidney.html")

@app.route('/kidneypredict',methods=['POST','GET'])
def kidenypredict():
    model2= pickle.load(open('modelkidney1.pkl', 'rb'))

    features2=[int(x) for x in request.form.values()]

    final2=[np.array(features2)]

    print(features2)

    print(final2)

    prediction2=model2.predict(final2)


    output =  prediction2

    if output==1:

        return render_template('rheart.html', pred='The patient is diagnosed with kidneyproblem. ')

    else:
        return render_template('rheart.html',pred='The patient is not diagnosed with kidneyproblem.')


@app.route('/liver')
def liver():
    return render_template("liver.html")

@app.route('/liverpredict',methods=['POST','GET'])
def liverpredict():
    model2= pickle.load(open('modelliver.pkl', 'rb'))

    features2=[int(x) for x in request.form.values()]

    final2=[np.array(features2)]

    print(features2)

    print(final2)

    prediction2=model2.predict(final2)


    output =  prediction2

    if output==1:

        return render_template('rliver.html', pred='The patient is diagnosed with liverproblem. ')

    else:
        return render_template('rliver.html',pred='The patient is not diagnosed with liverproblem.')





if __name__ == '__main__':
    app.run(debug=True)