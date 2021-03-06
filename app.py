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


@app.route('/cancerpredict',methods=['POST','GET'])
def cancerpredict():
    model2= pickle.load(open('modelcan.pkl', 'rb'))

    features2=[float(x) for x in request.form.values()]

    final2=[np.array(features2)]

    print(features2)

    print(final2)

    prediction2=model2.predict(final2)


    output =  prediction2

    if int(output)>=1:


        return render_template('rdiabetes.html', pred='The patient is diagnosed with Cancer. ')

    else:
        return render_template('rdiabetes.html',pred='The patient is not diagnosed with Cancer .')



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


@app.route('/kidney')
def kidney():
    return render_template("kidney.html")

@app.route('/kpredict',methods=['POST','GET'])
def kpredict():
    model2= pickle.load(open('modelkidney1.pkl', 'rb'))

    features2=[float(x) for x in request.form.values()]

    final2=[np.array(features2)]

    print(features2)

    print(final2)

    prediction2=model2.predict(final2)


    output =  prediction2
    print(output)

    if int(output)>=1:

        return render_template('rliver.html', pred='The patient is diagnosed with kidneyproblem. ')

    else:
        return render_template('rliver.html',pred='The patient is not diagnosed with kidneyproblem.')





if __name__ == '__main__':
    app.run(debug=True)