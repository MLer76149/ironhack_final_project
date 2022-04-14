import pandas as pd
import numpy as np
import json
import pickle


words_de = pd.read_csv("final_words/finalwords_clean_DE.csv")
words_de = words_de.drop(columns=["Unnamed: 0"])
words_ca = pd.read_csv("final_words/finalwords_clean_CA.csv")
words_ca = words_ca.drop(columns=["Unnamed: 0"])
words_us = pd.read_csv("final_words/finalwords_clean_US.csv")
words_us = words_us.drop(columns=["Unnamed: 0"])
words_fr = pd.read_csv("final_words/finalwords_clean_FR.csv")
words_fr = words_fr.drop(columns=["Unnamed: 0"])
words_in = pd.read_csv("final_words/finalwords_clean_IN.csv")
words_in = words_in.drop(columns=["Unnamed: 0"])
words_gb = pd.read_csv("final_words/finalwords_clean_GB.csv")
words_gb = words_gb.drop(columns=["Unnamed: 0"])


cat_de = pd.read_csv("plot_df/plot_cat_DE_DE.csv")
cat_de = cat_de.drop(columns=["Unnamed: 0"])
cat_choose_de = cat_de[~cat_de["count"].isna() & ~cat_de["category"].isna()]
cat_ca = pd.read_csv("plot_df/plot_cat_CA_EN.csv")
cat_ca = cat_ca.drop(columns=["Unnamed: 0"])
cat_choose_ca = cat_ca[~cat_ca["count"].isna() & ~cat_ca["category"].isna()]
cat_us = pd.read_csv("plot_df/plot_cat_US_EN.csv")
cat_us = cat_us.drop(columns=["Unnamed: 0"])
cat_choose_us = cat_us[~cat_us["count"].isna() & ~cat_us["category"].isna()]
cat_fr = pd.read_csv("plot_df/plot_cat_FR_FR.csv")
cat_fr = cat_fr.drop(columns=["Unnamed: 0"])
cat_choose_fr = cat_fr[~cat_fr["count"].isna() & ~cat_fr["category"].isna()]
cat_in = pd.read_csv("plot_df/plot_cat_IN_EN.csv")
cat_in = cat_in.drop(columns=["Unnamed: 0"])
cat_choose_in = cat_in[~cat_in["count"].isna() & ~cat_in["category"].isna()]
cat_gb = pd.read_csv("plot_df/plot_cat_GB_EN.csv")
cat_gb = cat_gb.drop(columns=["Unnamed: 0"])
cat_choose_gb = cat_gb[~cat_gb["count"].isna() & ~cat_gb["category"].isna()]


empty_de = pd.read_csv("final_df/empty_df_DE_DE.csv")
empty_ca = pd.read_csv("final_df/empty_df_CA_EN.csv")
empty_us = pd.read_csv("final_df/empty_df_US_EN.csv")
empty_fr = pd.read_csv("final_df/empty_df_FR_FR.csv")
empty_in = pd.read_csv("final_df/empty_df_IN_EN.csv")
empty_gb = pd.read_csv("final_df/empty_df_GB_EN.csv")


def get_words(lang):
    
    if lang == "Germany":
        return list(words_de.columns)
    
    elif lang == "Canada":
        return list(words_ca.columns)
    
    elif lang == "USA":
        return list(words_us.columns)
    
    elif lang == "France":
        return list(words_fr.columns)
    
    elif lang == "India":
        return list(words_in.columns)
    
    elif lang == "Great Britain":
        return list(words_gb.columns)
    
    
def get_categories(lang):
    
    if lang == "Germany":
        
        return list(cat_choose_de["category"])
    
    elif lang == "Canada":
        
        return list(cat_choose_ca["category"])
    
    elif lang == "USA":
        
        return list(cat_choose_us["category"])
    
    elif lang == "France":
        
        return list(cat_choose_fr["category"])
    
    elif lang == "India":
        
        return list(cat_choose_in["category"])
    
    elif lang == "Great Britain":
        
        return list(cat_choose_gb["category"])
    
    
def create_data_predict(keywords, lang, category, target):
        
        row_dict = {}
        rtwo = 0
        mae = 0
        
        y = 0
        
        if target == "Likes Ratio":
            target = "Likes_Rate"
        
        if len(keywords) == 0:
            
            return y, rtwo, mae            
    
        if lang == "Germany":
            
            with open("final_models/error_dict_DE_DE.json") as f:
                data = json.load(f)
            
            if target == "Likes":
                rtwo = data["likes"]["R2"]
                mae = data["likes"]["MAE"]
                
            elif target == "Views":
                rtwo = data["views"]["R2"]
                mae = data["views"]["MAE"]
                
            elif target == "Likes_Rate":
                rtwo = data["likes_rate"]["R2"]
                mae = data["likes_rate"]["MAE"]*100
            
            category_id = int(cat_choose_de[cat_choose_de["category"] == category]["id"])
            
            for i, keyword in enumerate(keywords):
                for col in empty_de.columns:
                    if keyword in col:
                        row_dict[col] = 1
                    elif i == 0:
                        row_dict[col] = 0
                        
            row_dict["category_id"] = category_id
            
            df_predict = pd.DataFrame.from_dict(row_dict, orient="index")
            df_predict = df_predict.T
            
            y = __predict(df_predict, lang, target)
            
            if target == "Likes" or target == "Views":
                y = int(y)
                
                return y, rtwo, mae
            
            return np.round(float(y), 2), rtwo, mae

        
        elif lang == "Canada":
            
            with open("final_models/error_dict_CA_EN.json") as f:
                data = json.load(f)
            
            if target == "Likes":
                rtwo = data["likes"]["R2"]
                mae = data["likes"]["MAE"]
                
            elif target == "Views":
                rtwo = data["views"]["R2"]
                mae = data["views"]["MAE"]
                
            elif target == "Likes_Rate":
                rtwo = data["likes_rate"]["R2"]
                mae = data["likes_rate"]["MAE"]*100
            
            category_id = int(cat_choose_ca[cat_choose_ca["category"] == category]["id"])
            
            for i, keyword in enumerate(keywords):
                for col in empty_de.columns:
                    if keyword in col:
                        row_dict[col] = 1
                    elif i == 0:
                        row_dict[col] = 0
                        
            row_dict["category_id"] = category_id
            
            df_predict = pd.DataFrame.from_dict(row_dict, orient="index")
            df_predict = df_predict.T
            
            y = __predict(df_predict, lang, target)
            
            if target == "Likes" or target == "Views":
                y = int(y)
                
                return y, rtwo, mae
            
            return np.round(float(y), 2),rtwo, mae
        
        
        elif lang == "USA":
            
            with open("final_models/error_dict_US_EN.json") as f:
                data = json.load(f)
            
            if target == "Likes":
                rtwo = data["likes"]["R2"]
                mae = data["likes"]["MAE"]
                
            elif target == "Views":
                rtwo = data["views"]["R2"]
                mae = data["views"]["MAE"]
                
            elif target == "Likes_Rate":
                rtwo = data["likes_rate"]["R2"]
                mae = data["likes_rate"]["MAE"]*100
            
            category_id = int(cat_choose_us[cat_choose_us["category"] == category]["id"])
            
            for i, keyword in enumerate(keywords):
                for col in empty_de.columns:
                    if keyword in col:
                        row_dict[col] = 1
                    elif i == 0:
                        row_dict[col] = 0
                        
            row_dict["category_id"] = category_id
            
            df_predict = pd.DataFrame.from_dict(row_dict, orient="index")
            df_predict = df_predict.T
            
            y = __predict(df_predict, lang, target)
            
            if target == "Likes" or target == "Views":
                y = int(y)
                
                return y, rtwo, mae
            
            return np.round(float(y), 2), rtwo, mae
        
        
        elif lang == "France":
            
            with open("final_models/error_dict_FR_FR.json") as f:
                data = json.load(f)
            
            if target == "Likes":
                rtwo = data["likes"]["R2"]
                mae = data["likes"]["MAE"]
                
            elif target == "Views":
                rtwo = data["views"]["R2"]
                mae = data["views"]["MAE"]
                
            elif target == "Likes_Rate":
                rtwo = data["likes_rate"]["R2"]
                mae = data["likes_rate"]["MAE"]*100
            
            category_id = int(cat_choose_fr[cat_choose_fr["category"] == category]["id"])
            
            for i, keyword in enumerate(keywords):
                for col in empty_de.columns:
                    if keyword in col:
                        row_dict[col] = 1
                    elif i == 0:
                        row_dict[col] = 0
                        
            row_dict["category_id"] = category_id
            
            df_predict = pd.DataFrame.from_dict(row_dict, orient="index")
            df_predict = df_predict.T
            
            y = __predict(df_predict, lang, target)
            
            if target == "Likes" or target == "Views":
                y = int(y)
                
                return y, rtwo, mae
            
            return np.round(float(y), 2), rtwo, mae
        
        
        elif lang == "India":
            
            with open("final_models/error_dict_IN_EN.json") as f:
                data = json.load(f)
            
            if target == "Likes":
                rtwo = data["likes"]["R2"]
                mae = data["likes"]["MAE"]
                
            elif target == "Views":
                rtwo = data["views"]["R2"]
                mae = data["views"]["MAE"]
                
            elif target == "Likes_Rate":
                rtwo = data["likes_rate"]["R2"]
                mae = data["likes_rate"]["MAE"]*100
            
            category_id = int(cat_choose_in[cat_choose_in["category"] == category]["id"])
            
            for i, keyword in enumerate(keywords):
                for col in empty_de.columns:
                    if keyword in col:
                        row_dict[col] = 1
                    elif i == 0:
                        row_dict[col] = 0
                        
            row_dict["category_id"] = category_id
            
            df_predict = pd.DataFrame.from_dict(row_dict, orient="index")
            df_predict = df_predict.T
            
            y = __predict(df_predict, lang, target)
            
            if target == "Likes" or target == "Views":
                y = int(y)
                
                return y, rtwo, mae
            
            return np.round(float(y), 2), rtwo, mae
        
        
        elif lang == "Great Britain":
            
            with open("final_models/error_dict_GB_EN.json") as f:
                data = json.load(f)
            
            if target == "Likes":
                rtwo = data["likes"]["R2"]
                mae = data["likes"]["MAE"]
                
            elif target == "Views":
                rtwo = data["views"]["R2"]
                mae = data["views"]["MAE"]
                
            elif target == "Likes_Rate":
                rtwo = data["likes_rate"]["R2"]
                mae = data["likes_rate"]["MAE"]*100
            
            category_id = int(cat_choose_gb[cat_choose_gb["category"] == category]["id"])
            
            for i, keyword in enumerate(keywords):
                for col in empty_de.columns:
                    if keyword in col:
                        row_dict[col] = 1
                    elif i == 0:
                        row_dict[col] = 0
                        
            row_dict["category_id"] = category_id
            
            df_predict = pd.DataFrame.from_dict(row_dict, orient="index")
            df_predict = df_predict.T
            
            y = __predict(df_predict, lang, target)
            
            if target == "Likes" or target == "Views":
                y = int(y)
                
                return y, rtwo, mae
            
            return np.round(float(y), 2), rtwo, mae
        

def __predict(df_predict, lang, target):
    
    if lang == "Germany":
    
        filename = "final_models/DE_DE_" + target + "_random_forest.sav"

        droplist = ["views", "dislikes", "comment_count", "likes", "likes_rate", "dislikes_rate"]

        df = df_predict.drop(columns=droplist)

        loaded_random_f = pickle.load(open(filename, 'rb'))

        y = loaded_random_f.predict(df)
        
        if target == "Likes_Rate":
            
            y = y * 100

        return y
    
    
    elif lang == "Canada":
    
        filename = "final_models/CA_EN_" + target + "_random_forest.sav"

        droplist = ["views", "dislikes", "comment_count", "likes", "likes_rate", "dislikes_rate"]

        df = df_predict.drop(columns=droplist)

        loaded_random_f = pickle.load(open(filename, 'rb'))

        y = loaded_random_f.predict(df)
        
        if target == "Likes_Rate":
            
            y = y * 100

        return y
    
    
    elif lang == "USA":
    
        filename = "final_models/US_EN_" + target + "_random_forest.sav"

        droplist = ["views", "dislikes", "comment_count", "likes", "likes_rate", "dislikes_rate"]

        df = df_predict.drop(columns=droplist)

        loaded_random_f = pickle.load(open(filename, 'rb'))

        y = loaded_random_f.predict(df)
        
        if target == "Likes_Rate":
            
            y = y * 100

        return y
    
    
    elif lang == "France":
    
        filename = "final_models/FR_FR_" + target + "_random_forest.sav"

        droplist = ["views", "dislikes", "comment_count", "likes", "likes_rate", "dislikes_rate"]

        df = df_predict.drop(columns=droplist)

        loaded_random_f = pickle.load(open(filename, 'rb'))

        y = loaded_random_f.predict(df)
        
        if target == "Likes_Rate":
            
            y = y * 100

        return y
    
    
    elif lang == "India":
    
        filename = "final_models/IN_EN_" + target + "_random_forest.sav"

        droplist = ["views", "dislikes", "comment_count", "likes", "likes_rate", "dislikes_rate"]

        df = df_predict.drop(columns=droplist)

        loaded_random_f = pickle.load(open(filename, 'rb'))

        y = loaded_random_f.predict(df)
        
        if target == "Likes_Rate":
            
            y = y * 100

        return y
    
    
    elif lang == "Great Britain":
    
        filename = "final_models/GB_EN_" + target + "_random_forest.sav"

        droplist = ["views", "dislikes", "comment_count", "likes", "likes_rate", "dislikes_rate"]

        df = df_predict.drop(columns=droplist)

        loaded_random_f = pickle.load(open(filename, 'rb'))

        y = loaded_random_f.predict(df)
        
        if target == "Likes_Rate":
            
            y = y * 100

        return y
    
    
    
    
    
    
    
        

    
    

        
        
    