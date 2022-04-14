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


def cat_cat(lang, lower_perc):
    
    bar_plot_text = ""
    bar_color = []
    
    if lang == "Canada":
        df = plot_ca.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "Canada"
        bar_color = ["hotpink"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
        
    elif lang == "Germany":
        
        df = plot_de.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "Germany"
        bar_color = ["dodgerblue"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
    elif lang == "France":
        
        df = plot_fr.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "France"
        bar_color = ["forestgreen"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
    elif lang == "Great Britain":
        
        df = plot_gb.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "Great Britain"
        bar_color = ["darkred"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
    elif lang == "India":
        
        df = plot_in.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "India"
        bar_color = ["peru"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
    elif lang == "USA":
        
        df = plot_us.copy()
        df = df[df["cat_percent"] > lower_perc]
        bar_plot_text = "USA"
        bar_color = ["indigo"]
        
        return df.sort_values(by="count", ascending=False), bar_plot_text, bar_color
    
def lang_lang(lang):
    
    bar_plot_text = ""
    bar_color = []
    
    if lang == "Canada":
        df = plot_ca_lang.copy()
        
        bar_plot_text = "Canada"
        bar_color = ["#f47258"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
        
    elif lang == "Germany":
        
        df = plot_de_lang.copy()
        
        bar_plot_text = "Germany"
        bar_color = ["dodgerblue"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
    
    elif lang == "France":
        
        df = plot_fr_lang.copy()
        
        bar_plot_text = "France"
        bar_color = ["forestgreen"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
    
    elif lang == "Great Britain":
        
        df = plot_gb_lang.copy()
        
        bar_plot_text = "Great Britain"
        bar_color = ["darkred"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
    
    elif lang == "India":
        
        df = plot_in_lang.copy()
        
        bar_plot_text = "India"
        bar_color = ["peru"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color
    
    elif lang == "USA":
        
        df = plot_us_lang.copy()
        
        bar_plot_text = "USA"
        bar_color = ["indigo"]
        
        return df.sort_values(by="percent", ascending=False), bar_plot_text, bar_color

