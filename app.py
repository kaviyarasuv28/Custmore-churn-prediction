#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[13]:


model = joblib.load('My_model.pkl')


# In[14]:


st.title("Customer Churn Predictoin")


# In[19]:


def user_input():
    account_length = st.sidebar.slider("Please Select Your account_lenght :",0,255)
    voice_mail_plan = st.sidebar.selectbox("Please Select Your voice mail plan details if no plan press 0 otherwise 1 :",[0,1])
    customer_service_calls = st.sidebar.selectbox("Please Enter How much times You Did call to Customer service :",[0,1,2,3,4])
    international_plan = st.sidebar.selectbox("Please Select Your International plan details if no plan press 0 otherwise 1 :",[0,1])
    total_charge = st.sidebar.slider("Please Enter how much Charge You payed Details :",0,100)
    total_calls = st.sidebar.slider("Please Enter how much Call You Did per dayDetails :",0,800)
    dict1={'account_length':account_length,'voice_mail_plan':voice_mail_plan,'customer_service_calls':customer_service_calls,'international_plan':international_plan,'total_charge':total_charge,'total_calls':total_calls}
    return pd.DataFrame(dict1,index=[0])
    
df =user_input()
if st.button("Predict"):
    proba = model.predict_proba(df)[:,1][0]
    threshold=0.5
    prediction=1 if proba >= threshold else 0
    st.subheader('Predicted Result')
    st.write('Churn' if prediction == 1 else 'No Churn')
    st.subheader('Predicted_proba')
    st.write(f"{proba:2f}")


# In[20]:


df


# In[ ]:





# In[ ]:




