from flask import Flask,render_template,request,url_for
from flask import jsonify
import flask_excel as excel
import requests
import pickle
import numpy as np
import pandas as pd
import sklearn
from werkzeug.utils import redirect

app=Flask(__name__)
model1=pickle.load(open('gssn_nb.pkl','rb'))
model2=pickle.load(open('brnl_nb.pkl','rb'))

def get_data():
    global df
    excel.init_excel(app)
    f = 'names.csv'
    data_xls = pd.read_csv(f)
    df=data_xls.sample(n=10)
    return df

df=get_data()


app=Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    xstr1=""
    for i,val in enumerate(df.name):
        xstr1+=val+" "
            
    return render_template('index1.html',result=xstr1.split())
    
    
def ascii_mean(name):
    ascii_list=[ord(x) for x in name]
    return np.array(ascii_list).mean()
def letter_class(name):
    letter_list=[x for x in name]
    vowel=0
    consonent=0
    for letter in letter_list:
        if letter in ['a','e','i','o','u']:
            vowel +=1
        else:
            consonent +=1
    return vowel,consonent

def name_convertor(name_list=df.name):
    ndf=pd.DataFrame([],columns=['ascii_value','name_len','num_vowels','num_consonent','last_letter_vowel'])
    ndf['name']=name_list
    ndf['ascii_value']=ndf['name'].apply(lambda x: ascii_mean(x).round(3))
    ndf['name_len']=ndf['name'].apply(lambda x: len(x))
    ndf['num_vowels']=ndf['name'].apply(lambda x: letter_class(x)[0])
    ndf['num_consonent']=ndf['name'].apply(lambda x: letter_class(x)[1])
    ndf['last_letter_vowel']=ndf['name'].apply(lambda x: 1 if x[-1] in ['a','e','i','o','u'] else 0)
        
    return ndf
    
   
@app.route('/success/<int:score>')
def success(score):
    res=""
    if score==0:
        res="Match Ties"
    elif score==1:
        res="Machine wins"
    else:
        res="You win"
    exp={'player':score,'res':res}
    return render_template('result1.html',result=exp)


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        ans1=bool(request.form['name_1'])
        if(ans1=='Yes'):
            ans1=1
        else:
            ans1=0
        ans2=bool(request.form['name_2'])
        if(ans2=='Yes'):
            ans2=1
        else:
            ans2=0
        ans3=bool(request.form['name_3'])
        if(ans3=='Yes'):
            ans3=1
        else:
            ans3=0
        ans4=bool(request.form['name_4'])
        if(ans4=='Yes'):
            ans4=1
        else:
            ans4=0
        ans5=bool(request.form['name_5'])
        if(ans5=='Yes'):
            ans5=1
        else:
            ans5=0
        ans6=bool(request.form['name_6'])
        if(ans6=='Yes'):
            ans6=1
        else:
            ans6=0
        ans7=bool(request.form['name_7'])
        if(ans7=='Yes'):
            ans7=1
        else:
            ans7=0
        ans8=bool(request.form['name_8'])
        if(ans8=='Yes'):
            ans8=1
        else:
            ans8=0
        ans9=bool(request.form['name_9'])
        if(ans9=='Yes'):
            ans9=1
        else:
            ans9=0
        ans10=bool(request.form['name_10'])
        if(ans10=='Yes'):
            ans10=1
        else:
            ans10=0
        my_guess=[ans1,ans2,ans3,ans4,ans5,ans6,ans7,ans8,ans9,ans10]
        
        d1=name_convertor(df.name)
        continuous_features=['ascii_value', 'name_len', 'num_vowels', 'num_consonent']
        categorical_features=['last_letter_vowel']
        pred1=model1.predict_proba(d1[continuous_features])
        pred2=model2.predict_proba(d1[categorical_features])
        final1=pred1*pred2
        predicts=final1[:,1]/np.sum(final1,axis=1)
        y_pred=list(map(lambda x:1 if x>0.65 else 0,predicts))
        
        d2=df.gender
        d2=d2.apply(lambda x: 1 if x=='Female' else 0)
        
        chk=[]
        chk1=0
        chk2=0
        for i in range(len(d2)):
            if y_pred[i]==d2.iloc[i]:
                chk1+=1
            elif my_guess[i]==d2.iloc[i]:
                chk2+=1
        chk.append(chk1)
        chk.append(chk2)
        if chk[0]==chk[1]:
            prediction=0
        elif chk[0]>chk[1]:
            prediction=1
        else:
            prediction=2
        return redirect(url_for('success',score=prediction))


if __name__=="__main__":
    app.run(debug=True)