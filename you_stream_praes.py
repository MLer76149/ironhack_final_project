import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import you_praes as yp
import you_plot as yt

# setting config of page
st.set_page_config(
    page_title="Youtube NLP",
    page_icon="🛸")

st.title(" 🛸 Youtube Keywords Analysis")


# showing categories
option = st.selectbox(
         'Category by Country',
         ('Start', 'Germany', 'India', 'Great Britain', 'Canada', 'USA', 'France'), )

if option != "Start":

    values = st.slider(
     'Select a percentage of categories',
     0.0, 100.0, (5.0, 75.0))
    st.write('Values:', values[0])

    tickangle = 0

    if values[0] < 5:
        tickangle = 45
    else:
        tickangle = 0

    df, plot_text, bar_color = yp.cat_cat(option, values[0])

    fig = px.bar(df, x="category", y="count",  hover_data = df, color_discrete_sequence=bar_color)
    fig.update_layout(
    {"updatemenus":[
        go.layout.Updatemenu()],
        'title_text': plot_text,
        'xaxis': dict(title='Category', tickangle=tickangle, tickfont=dict(size = 20)),
        'yaxis_title_text': 'Videos per Category',
        "width": 1200,
        "height": 700,
        "autosize": True,})

    st.plotly_chart(fig, use_container_width=True)
    
# showing languages        
option2 = st.selectbox(
         'Languages by Country',
         ('Start', 'Germany', 'India', 'Great Britain', 'Canada', 'USA', 'France'), )


if option2 != "Start":

    df, plot_text, bar_color = yp.lang_lang(option2)

    fig = px.bar(df, x="country", y="percent",  hover_data = df, color_discrete_sequence=bar_color)
    fig.update_layout(
    {"updatemenus":[
        go.layout.Updatemenu()],
        'title_text': plot_text,
        'xaxis': dict(title='Languages', tickangle=0, tickfont=dict(size = 20)),
        'yaxis_title_text': '%',
        "width": 1200,
        "height": 700,
        "autosize": True,})

    st.plotly_chart(fig, use_container_width=True)
    
# showing wordclouds        
option3 = st.selectbox(
         'Words by Country',
         ('Start', 'Germany', 'India', 'Great Britain', 'Canada', 'USA', 'France', 'All'), )


if option3 != "Start":
        
    image = ""

    if option3 == "Germany":
        image = "images_wordcloud/germany_cloud.jpg"

    elif option3 == "France":
        image = "images_wordcloud/france_cloud.jpg"

    elif option3 == "Canada":
        image = "images_wordcloud/canada_cloud.jpg"

    elif option3 == "USA":
        image = "images_wordcloud/usa_cloud.jpg"

    elif option3 == "India":
        image = "images_wordcloud/india_cloud.jpg"

    elif option3 == "Great Britain":
        image = "images_wordcloud/england_cloud.jpg"

    elif option3 == "All":
        image = "images_wordcloud/stormtrooper_cloud.jpg"

    col1, col2, col3 = st.columns(3)

    with col2:   
        st.image(image, width = 800)

            
    

        










        