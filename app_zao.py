#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np 
import pandas as pd 
import shap
import joblib
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn import svm
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[2]:


#import plotly.figure_factory as ff
import matplotlib.pyplot as plt


# In[3]:


plt.style.use('default')


# In[4]:


from sklearn.metrics import accuracy_score


# In[5]:


st.markdown("<h1 style='text-align: center; color: black;'>Online Interpretable Machine Learning Models for Dynamic Prediction of In-Hospital Infection in Patients after Heart Transplantation (early prediction model)</h1>", unsafe_allow_html=True)


# In[ ]:


# side-bar 
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.number_input("Age")
    a2 = st.sidebar.number_input("cardiopulmonary bypass duration (min)")
    a3 = st.sidebar.number_input("surgery duration (min)")
    a4 = st.sidebar.number_input("preoperative hemoglobin (g/L)")
    a5 = st.sidebar.number_input("preoperative aspartate aminotransferase（U/L）")
    
    output = [a1,a2,a3,a4,a5]
    return output

outputdf = user_input_features()


# In[ ]:


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    xgb = joblib.load("early.m")
    #标准化
    a1 = outputdf[0]
    a2 = outputdf[1]
    a3 = outputdf[2]
    a4 = outputdf[3]
    a5 = outputdf[4]
    # Store inputs into dataframe
    a1s=(a1-52.32608696)/11.39136658
    a2s=(a2-173.7820624)/37.21589971
    a3s=(a3-320.8983357)/66.94806709
    a4s=(a4-135.7640451)/23.75809376
    a5s=(a5-66.65857641)/144.5900799
    stdf = [a1s,a2s,a3s,a4s,a5s]
    
    X = pd.DataFrame([outputdf], columns= ["Age", "术中体外循环时间min", "手术时间","血红蛋白0","门冬氨酸氨基转移酶0"])
    X_standard = pd.DataFrame([stdf], columns= ["Age", "术中体外循环时间min", "手术时间","血红蛋白0","门冬氨酸氨基转移酶0"])
    # Get prediction
    p1 = xgb.predict(X_standard)[0]
    p2 = xgb.predict_proba(X_standard)[0,-1]
    m1 = round(float(p2) * 100, 2)
    p3 = "%.2f%%" % (m1)

    # Output prediction
    st.write(f'Predicted results: {p1}')
    st.write('0️⃣ means non-infection, 1️⃣ means infection')
    st.text(f"Prediction probabilities：1️⃣ {p3}")
    
    #SHAP
    st.title('SHAP')
    
    #个例
    #X_standard.columns = ['Age','cardiopulmonary bypass duration','surgical time','preoperative hemoglobin','preoperative aspartate aminotransferase']
    X.columns = ['Age','CPB duration','surgical time',
                              'preoperative HB','preoperative AST']
    explainer_xgb = shap.TreeExplainer(xgb)
    shap_values= explainer_xgb.shap_values(X_standard)
    #shap.initjs()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.force_plot(explainer_xgb.expected_value, shap_values[0],X.iloc[0],link='logit',matplotlib=True)

    #shap瀑布图
    #shap_values2 = explainer_xgb(X_standard) 
    #shap.plots.waterfall(shap_values2[0])
    st.pyplot(bbox_inches='tight')
    

# In[ ]:

