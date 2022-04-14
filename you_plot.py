import pandas as pd
import numpy as np
import json



def create_category_dict(country, lang):
    
    df = pd.read_csv("final_df/finaldf_"+country+".csv")
    
    cat_dict = dict(df["category_id"].value_counts())
        
    df["category_count"] = 0
    
    for i, item in enumerate(df["category_id"]):
        for key in cat_dict:
            if key == item:
                df.loc[i, "category_count"] = cat_dict[key]
                
    plot_df = pd.DataFrame([df["category_count"].unique(), df["category_id"].unique()]).T
    plot_df.columns = ["count", "id"]
    
    with open("youtube_raw_data/"+lang+"_category_id.json") as f:
        data = json.load(f)
    
    cat_id_dict = {}

    for item in data["items"]:
        cat_id_dict[item["id"]] = item["snippet"]["title"]
        
    plot_df["category"] = ""
    for i, item in enumerate(plot_df["id"]):
        for key in cat_id_dict:
            if key == str(int(item)):
                plot_df.loc[i, "category"] = cat_id_dict[key]
    
    plot_df["cat_percent"] = np.round(plot_df["count"]/plot_df["count"].sum() * 100, 2)
                
    plot_df.to_csv("plot_df/plot_cat_" + country + ".csv")
    
    return plot_df


def create_language_dict(country):
    
    df = pd.read_csv("youtube_final_cleaned_data/"+country+"videos_final_cleaned.csv")
    
    df["raw_language"] = df["raw_language"].fillna("unknown")
        
    langu = pd.DataFrame.from_dict(dict(df["raw_language"].value_counts()), orient="index")
    
    langu["percent"] = np.round(langu[0] / langu[0].sum() * 100, 2)
                
    langu = langu[langu["percent"] > 2]
    
    langu["country"] = langu.index
    
    langu = langu.reset_index(drop=True)
    
    langu = langu.drop(columns=[0])
                        
    langu.to_csv("plot_df/plot_lang_" + country + ".csv")
    
    return langu


def create_empty_df(country):
    
    df = pd.read_csv("final_df/finaldf_"+country+".csv")
    
    df = df.select_dtypes(np.number)
    
    empt_df = pd.DataFrame(columns=df.columns)
                
    empt_df.to_csv("final_df/empty_df_" + country + ".csv", index=False)
    
    return empt_df


def create_error_dict(country):
    
    error_dict = {"likes": {"R2": 0,
                          "MAE": 0
                          },
                "likes_rate": {
                          "R2": 0,
                          "MAE": 0
                            },
                 "views": {"R2": 0,
                          "MAE": 0}
                            }
    
    likes_r2 = float(input("Likes R2: "))
    likes_mae = float(input("Likes MAE: "))
    likes_rate_r2 = float(input("Likes_Rate R2: "))
    likes_rate_mae = float(input("Likes_Rate MAE: "))
    views_r2 = float(input("Views R2: "))
    views_mae = float(input("Views MAE: "))
    
    error_dict["likes"]["R2"] = np.round(likes_r2, 2)
    error_dict["likes"]["MAE"] = np.round(likes_mae, 2)
    error_dict["likes_rate"]["R2"] = np.round(likes_rate_r2, 2)
    error_dict["likes_rate"]["MAE"] = np.round(likes_rate_mae, 2)
    error_dict["views"]["R2"] = np.round(views_r2, 2)
    error_dict["views"]["MAE"] = np.round(views_mae, 2)
    
    with open('final_models/error_dict_'+country+'.json', 'w') as fp:
        json.dump(error_dict, fp)

    