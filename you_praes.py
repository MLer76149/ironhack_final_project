import pandas as pd
import numpy as np


plot_ca = pd.read_csv("plot_df/plot_cat_CA_EN.csv")
plot_de = pd.read_csv("plot_df/plot_cat_DE_DE.csv")
plot_fr = pd.read_csv("plot_df/plot_cat_FR_FR.csv")
plot_gb = pd.read_csv("plot_df/plot_cat_GB_EN.csv")
plot_in = pd.read_csv("plot_df/plot_cat_IN_EN.csv")
plot_us = pd.read_csv("plot_df/plot_cat_US_EN.csv")

plot_ca_lang = pd.read_csv("plot_df/plot_lang_CA.csv")
plot_de_lang = pd.read_csv("plot_df/plot_lang_DE.csv")
plot_fr_lang = pd.read_csv("plot_df/plot_lang_FR.csv")
plot_gb_lang = pd.read_csv("plot_df/plot_lang_GB.csv")
plot_in_lang = pd.read_csv("plot_df/plot_lang_IN.csv")
plot_us_lang = pd.read_csv("plot_df/plot_lang_US.csv")

views = pd.read_csv("plot_df/views.csv")
views = views.iloc[0:25,:]
likes = pd.read_csv("plot_df/likes.csv")
likes = likes.iloc[0:25,:]
likes_ratio = pd.read_csv("plot_df/likes_rate.csv")
likes_ratio = likes_ratio.iloc[0:25,:]

def get_df():

    return views, likes, likes_ratio


def cat_cat(lang, lower_perc):
    
    bar_plot_text = ""
    bar_color = []
    
    if lang == "Canada":
        df = plot_ca.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "Canada"
        bar_color = ["#4FAAA1"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
        
    elif lang == "Germany":
        
        df = plot_de.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "Germany"
        bar_color = ["#FF8B8B"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
    elif lang == "France":
        
        df = plot_fr.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "France"
        bar_color = ["#CC759A"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
    elif lang == "United Kingdom":
        
        df = plot_gb.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "United Kingdom"
        bar_color = ["#A09FE6"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
    elif lang == "India":
        
        df = plot_in.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "India"
        bar_color = ["#B8F4FF"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
    elif lang == "USA":
        
        df = plot_us.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "USA"
        bar_color = ["#FFB3BA"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
def lang_lang(lang):
    
    bar_plot_text = ""
    bar_color = []
    
    if lang == "Canada":
        df = plot_ca_lang.copy()
        
        bar_plot_text = "Canada"
        bar_color = ["#4FAAA1"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
        
    elif lang == "Germany":
        
        df = plot_de_lang.copy()
        
        bar_plot_text = "Germany"
        bar_color = ["#FF8B8B"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
    
    elif lang == "France":
        
        df = plot_fr_lang.copy()
        
        bar_plot_text = "France"
        bar_color = ["#CC759A"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
    
    elif lang == "United Kingdom":
        
        df = plot_gb_lang.copy()
        
        bar_plot_text = "United Kingdom"
        bar_color = ["#A09FE6"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
    
    elif lang == "India":
        
        df = plot_in_lang.copy()
        
        bar_plot_text = "India"
        bar_color = ["#B8F4FF"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
    
    elif lang == "USA":
        
        df = plot_us_lang.copy()
        
        bar_plot_text = "USA"
        bar_color = ["#FFB3BA"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color

