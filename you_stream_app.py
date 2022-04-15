import pandas as pd
import numpy as np
import streamlit as st
import you_app as ya



st.set_page_config(
    page_title="Youtube NLP",
    page_icon="🛸")

st.title(" 🛸 Youtube Keywords Prediction")

country = st.selectbox(
         'Country',
         ('Start', 'Germany', 'India', 'United Kingdom', 'Canada', 'USA', 'France'), )

# get keywords by country
key_words = ya.get_words(country)

if country != "Start":

    keywords = st.multiselect(
         'Please choose your words',
         key_words)
    
    words = ""
    
    for i in range(len(keywords)):
        
        if i < len(keywords) - 1:                   
            words += keywords[i] + ", "
        
        else:
            words += keywords[i]

    st.header('You selected: '+ words)
    
    
# get categories
categories = ya.get_categories(country)

if country != "Start":

    category = st.selectbox(
         'Please choose your category',
         categories)
    
if country != "Start":
    
    target = st.radio(
     "Which target do you want to predict?",
     ('Likes', 'Likes Ratio', 'Views'))

    y, r2, mae = ya.create_data_predict(keywords, country, category, target)
    st.header("Your prediction: " + str(y))
    
    st.write('R2:', r2)
    st.write('MAE:', mae)

    

