import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import you_praes as yp
import you_plot as yt

views, likes, likes_ratio = yp.get_df()


# setting config of page
st.set_page_config(
    page_title="Youtube NLP",
    page_icon="🛸",
    layout="wide",
)

st.title(" 🛸 Youtube Keywords Analysis")

with st.expander("Intro"):
    
    st.image("images_wordcloud/youtube_black.png", width = 150)
    
    st.markdown(""" # Trending Youtube Video Statistics 
    
    #### Funfacts:
    
    - Kenyan javelin thrower Julius Yego, who won Olympic silver in 2016, learned how to throw properly by watching YouTube videos.
    - When Google built a neural network and connected it to YouTube, it spontaneously taught itself to recognize cats with 74.8% accuracy.
    - YouTube was supposed to be a dating website, but destiny had other plans. We aren’t complaining!
    
    #### Old Days 2005:

   """)
    
    st.image("images_wordcloud/youtube_old_days.png")

    st.markdown(""" 
    
    #### Datasource: Kaggle

    #### Years: 2017 - 2018

    YouTube (the world-famous video sharing website) maintains a list of the top trending videos on the platform.\n 
    According to Variety magazine, “To determine the year’s top-trending videos, YouTube uses a combination of factors including measuring users 
    interactions (number of views, shares, comments and likes).

    This dataset is a daily record of the top trending YouTube videos.

    Around 50.000 unique records each dataset
    
    I used:
    #### United Kingdom, Germany, Canada, France, India, USA
    
    I didn't use: 
    #### Japan, Korea, Mexico, Russia """)
    
with st.expander("Data processing"):
    st.markdown(""" 
    
    #### Target: Likes, Likes Ratio, Views
    
    - language recognition
    - clean text with regex
    - nltk: stopwords and Lemmatizer
    
    #### Topwords: 50 of each Dataset
    
    #### ML
    
    - Random Forest
    - KNN
    - Linear Regression \n
    
    
    
    #### Trained 126 models in total
    
    

 """)
    
    col4, col5 = st.columns([1, 1])
    
    with col4:
        st.subheader("Likes")       
    
        st.dataframe(likes)
    
    with col5:
        st.subheader("Likes Ratio")    

        st.dataframe(likes_ratio)

    

with st.expander("Visualization"):
    
# showing categories
    option = st.selectbox(
             'Category by Country',
             ('Start', 'Germany', 'India', 'United Kingdom', 'Canada', 'USA', 'France'), )

    if option != "Start":

        values = st.slider(
         'Select a percentage of categories',
         0.0, 100.0, (5.0, 75.0))

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
             ('Start', 'Germany', 'India', 'United Kingdom', 'Canada', 'USA', 'France'), )


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
             ('Start', 'Germany', 'India', 'United Kingdom', 'Canada', 'USA', 'France', 'All'), )


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

        elif option3 == "United Kingdom":
            image = "images_wordcloud/england_cloud.jpg"

        elif option3 == "All":
            image = "images_wordcloud/stormtrooper_cloud.jpg"

        col1, col2, col3 = st.columns([1,4,1])

        with col1:

            if option3 == "Germany":
                st.title("🇩🇪")

            elif option3 == "France":
                st.title("🇫🇷")

            elif option3 == "Canada":
                st.title("🇨🇦")

            elif option3 == "USA":
                st.title("🇺🇸")

            elif option3 == "India":
                st.title("🇮🇳")

            elif option3 == "United Kingdom":
                st.title("🇬🇧")

            elif option3 == "All":
                st.title("👽")


        with col2:   
            st.image(image, width = 800)
            
with st.expander("Good Bye"):
    
    st.header("Thanks a lot!!!")
    
    st.image("images_wordcloud/darth.png", width = 150)
    
    st.header("Questions???")
    
    
    
    
        
      

            
    

        










        