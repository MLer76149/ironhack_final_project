import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as __plt
import seaborn as __sns
import math as __mt
import pickle
from google.colab import drive
from nltk.stem import WordNetLemmatizer



# get the key words

def get_key_words(df, language):
    
    lem = WordNetLemmatizer()
    
    data_nlp = df[df["raw_language"] == language]
    
    text_content = ""

    for text in data_nlp['cleaned_text']:
        text_content += text
        
    corpus = text_content.split()
    
    wordfreq = {}
    for word in corpus:
            if ( word not in wordfreq.keys() ):
                wordfreq[word] = 1 
            else:
                wordfreq[word] += 1
                
    freq_lem = {}
    
    for key in wordfreq:
        freq_lem[lem.lemmatize(key)] = wordfreq[key]
        
    return data_nlp, freq_lem


def create_keywords_df(df, wordfreq, stop_words, dataname):
    
    cols, corpus_freq = __processing_stopwords(wordfreq, stop_words)
        
    my_list = list( map(__review_inpector, df['cleaned_text'], 
                        [stop_words]*df.shape[0], [list(cols.keys())]*df.shape[0] ) )
    keywords_df = pd.DataFrame(my_list)
    
    final_df = pd.concat([df.reset_index(drop=True), keywords_df], axis=1)
    
    freq_df = pd.DataFrame(corpus_freq)
    freq_df.to_csv("/content/drive/MyDrive/Colab Notebooks/ironhack-final-project/final_words/finalwords_"+dataname+".csv", index=False)
    final_df.to_csv("/content/drive/MyDrive/Colab Notebooks/ironhack-final-project/final_df/finaldf_"+dataname+".csv", index=False)
    
    return final_df, corpus_freq


def __processing_stopwords(wordfreq, stop_words):
    
    for i in range(len(stop_words)):
        stop_words[i] = re.sub(r"\s*'\s*\w*","",stop_words[i])
        
    corpus = [(wordfreq[key],key) for key in list(wordfreq.keys()) if key not in stop_words and len(key) > 1]
    
    corpus.sort(reverse = True)
    
    corpus_freq = [(word[1],word[0]) for word in corpus[:51]] 
    
    cols = {word[0]: [] for word in corpus_freq}
    
    return cols, corpus_freq
  

def __review_inpector(sentence, stop_words, words):
    
    lem = WordNetLemmatizer()

    tokens = sentence.split()
    
    for i in range(len(tokens)):
        tokens[i] = lem.lemmatize(tokens[i])

    tokens = [ token for token in tokens if (token not in stop_words and token != '')]

    # Initializing an empty dictionary of word frequencies for the corresponding review
    col_freq = {col:0 for col in words}
    
    # Filling the dictionary with word frequencies in the review
    for token in tokens:
        if token in words:
            col_freq[token] += 1

    return col_freq


def train_model(df, target, filename):
    
    df = df.copy()
    df = df.select_dtypes(np.number)
    droplist = ["views", "dislikes", "comment_count", "likes", "likes_rate", "dislikes_rate"]
    droplist.remove(target)
    df = df.drop(columns=droplist)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test  = pd.DataFrame(X_test, columns=X.columns)

    y_train = pd.DataFrame(y_train, columns =[target])
    y_test  = pd.DataFrame(y_test, columns =[target])
    
    filenames = linear_regression(X_test, y_test, filename, X_train = X_train, y_train = y_train)
    
    return X_train, y_train, X_test, y_test, filenames
    

def linear_regression(X_test, y_test, filename, X_train = None, y_train = None, train = True):
    
    drive.mount("/content/drive")

    #X_train:
    ptransformer = PowerTransformer()
    ptransformer.fit(X_train) 
    X_train_ptrans = ptransformer.transform(X_train)
    X_train_ptrans_df = pd.DataFrame(X_train_ptrans, columns=X_train.columns, index = X_train.index )
    
    #X_test:
    X_test_ptrans = ptransformer.transform(X_test)
    X_test_ptrans_df = pd.DataFrame(X_test_ptrans, columns=X_test.columns, index = X_test.index )
       
    filename_pt = filename + "_power_transform.sav"
    pickle.dump(ptransformer, open("/content/drive/MyDrive/Colab Notebooks/ironhack-final-project/transformer_scaler/"+filename_pt, 'wb'))
    
    scaler = StandardScaler()
    scaler.fit(X_train_ptrans_df)
    
    filename_sc = filename + "_standard_scaler.sav"
    pickle.dump(scaler, open("/content/drive/MyDrive/Colab Notebooks/ironhack-final-project/transformer_scaler/"+filename_sc, 'wb'))
    
    X_scaled_train = scaler.transform(X_train_ptrans_df)
    X_train_ptrans_df = pd.DataFrame(X_scaled_train, columns=X_train_ptrans_df.columns)
    
    X_scaled_test = scaler.transform(X_test_ptrans_df)
    X_test_ptrans_df = pd.DataFrame(X_scaled_test, columns=X_test_ptrans_df.columns)

    
    if train:
        knn_models = __search_k(X_train_ptrans_df, y_train,X_test_ptrans_df, y_test)
        var = int(input("Please enter k:"))
        files = []
        
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)
        filename_rf = filename + "_random_forest.sav"
        pickle.dump(rf, open("/content/drive/MyDrive/Colab Notebooks/ironhack-final-project/models/"+filename_rf, 'wb'))
        print("-----------------------------")
        print("--------Random Forest--------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("-----------------------------")
        print("--------Random Forest--------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        print("Filename Linear: " + filename_rf)
        files.append(filename_rf)
        
        lr = LinearRegression()
        lr.fit(X_train_ptrans_df, y_train)
        y_pred_train = lr.predict(X_train_ptrans_df)
        y_pred_test = lr.predict(X_test_ptrans_df)
        filename_lr = filename + "_linear.sav"
        pickle.dump(lr, open("/content/drive/MyDrive/Colab Notebooks/ironhack-final-project/models/"+filename_lr, 'wb'))
        print("-----------------------------")
        print("------Linear Regression------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("-----------------------------")
        print("------Linear Regression------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        print("Filename Linear: " + filename_lr)
        files.append(filename_lr)
        
        knn_models[var-2].score(X_test_ptrans_df, y_test)
        y_pred_train = knn_models[var-2].predict(X_train_ptrans_df)
        y_pred_test = knn_models[var-2].predict(X_test_ptrans_df)
        filename_knn = filename + "_knn.sav"
        pickle.dump(knn_models[var-2], open("/content/drive/MyDrive/Colab Notebooks/ironhack-final-project/models/"+filename_knn, 'wb'))
        print("--------------KNN------------")
        print("----------Train Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_train , y_pred_train))
        print("MSE:",mean_squared_error(y_train , y_pred_train))
        print("RMSE:",np.sqrt(mean_squared_error(y_train , y_pred_train)))
        print("MAE:",mean_absolute_error(y_train , y_pred_train))
        print("Filename knn: " + filename_knn)
        print("-----------------------------")
        print("--------------KNN------------")
        print("-----------Test Set----------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred_test))
        print("MSE:",mean_squared_error(y_test , y_pred_test))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred_test)))
        print("MAE:",mean_absolute_error(y_test , y_pred_test))
        print("-----------------------------")
        files.append(filename_knn)
        
        return files
    
    if train == False:
        
        loaded_random_f = pickle.load(open("models/"+filename[0], 'rb'))
        y_pred1 = loaded_random_f.predict(X_test)
        print("--------Random Forest--------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred1))
        print("MSE:",mean_squared_error(y_test , y_pred1))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred1)))
        print("MAE:",mean_absolute_error(y_test , y_pred1))
        print("-----------------------------")
        
        loaded_linear = pickle.load(open("models/"+filename[1], 'rb'))
        y_pred2 = loaded_linear.predict(X_test)
        print("------Linear Regression------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred2))
        print("MSE:",mean_squared_error(y_test , y_pred2))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred2)))
        print("MAE:",mean_absolute_error(y_test , y_pred2))
        print("-----------------------------")
        
        loaded_knn = pickle.load(open("models/"+filename[2], 'rb'))
        y_pred3 = loaded_knn.predict(X_test)
        print("--------------KNN------------")
        print("-----------------------------")
        print("R2:",r2_score(y_test , y_pred3))
        print("MSE:",mean_squared_error(y_test , y_pred3))
        print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred3)))
        print("MAE:",mean_absolute_error(y_test , y_pred3))
        print("-----------------------------")
        
        return y_pred1, y_pred2, y_pred3

def __search_k(X_train, y_train, X_test, y_test):
    knn_models = []
    scores = []
    for k in range(2,15):
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_train, y_train)
        knn_models.append(model)
        scores.append(model.score(X_test, y_test))
    __plt.figure(figsize=(10,6))
    __plt.plot(range(2,15),scores,color = 'blue', linestyle='dashed',
    marker='o', markerfacecolor='red', markersize=10)
    __plt.title('R2-scores vs. K Value')
    __plt.xticks(range(1,16))
    __plt.gca().invert_yaxis()
    __plt.xlabel('K')
    __plt.ylabel('Accuracy')
    __plt.show()
    return knn_models

def random_search(X_train, y_train, random_grid, cv = 5, n_iter = 25):
    model = RandomForestRegressor()
    random_search = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter=n_iter, cv = cv, n_jobs = 2)
    random_search.fit(X_train,y_train)
    print(random_search.best_params_)
    print("The best R2 for the best hyperparameters is {:.2f}".format(random_search.best_score_))

    return random_search.best_params_

def random_forest_regr(X_test, y_test, forest_grid, filename, X_train = None, y_train = None):
    
    rf = RandomForestRegressor(max_depth=forest_grid['max_depth'],
                                 min_samples_split=forest_grid['min_samples_split'],
                                 min_samples_leaf =forest_grid['min_samples_leaf'],
                                 max_samples=forest_grid['max_samples'],
                                 random_state = forest_grid['random_state'])

    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    filename_rf = filename + "_random_forest_grid.sav"
    pickle.dump(rf, open("/content/drive/MyDrive/Colab Notebooks/ironhack-final-project/models/"+filename_rf, 'wb'))
    print("-----------------------------")
    print("--------Random Forest--------")
    print("----------Train Set----------")
    print("-----------------------------")
    print("R2:",r2_score(y_train , y_pred_train))
    print("MSE:",mean_squared_error(y_train , y_pred_train))
    print("RMSE:",np.sqrt(mean_squared_error(y_train , y_pred_train)))
    print("MAE:",mean_absolute_error(y_train , y_pred_train))
    print("-----------------------------")
    print("--------Random Forest--------")
    print("-----------Test Set----------")
    print("-----------------------------")
    print("R2:",r2_score(y_test , y_pred_test))
    print("MSE:",mean_squared_error(y_test , y_pred_test))
    print("RMSE:",np.sqrt(mean_squared_error(y_test , y_pred_test)))
    print("MAE:",mean_absolute_error(y_test , y_pred_test))
    print("-----------------------------")
    print("Filename Linear: " + filename_rf)
    
    return filename_rf


def boxplot_continous(df):
    
    r = __mt.ceil(df.shape[1]/2)
    c = 2
    fig, ax = __plt.subplots(r,c, figsize=(15,20))
    i = 0
    j = 0

    for item in df.columns:
        __sns.boxplot(x=item, data=df, ax=ax[i, j])
        if j == 0:
            j = 1
        elif j == 1:
            j = 0
            i = i + 1
    __plt.show() 
    
def remove_outliers(df):
    rem_df = df.copy()
    df_num = rem_df.select_dtypes(np.number)
    #df_cat = df.select_dtypes(__np.number)
    #df_other = df.select_dtypes(exclude=[__np.object, __np.number])

    old_rows = df.shape[0]
    for item in df_num.columns:
        print(item)
        iqr = np.nanpercentile(df[item],75) - np.nanpercentile(df[item],25)
        if iqr > 0:
            print(iqr)
            upper_limit = np.nanpercentile(df[item],75) + 1.5*iqr
            print(upper_limit)
            lower_limit = np.nanpercentile(df[item],25) - 1.5*iqr
            print(lower_limit)
            rem_df = rem_df[(rem_df[item] < upper_limit) & (rem_df[item] > lower_limit)]
        
    rows_removed = old_rows - df.shape[0]
    rows_removed_percent = (rows_removed/old_rows)*100
        
    print("{} rows have been removed, {}% in total".format(rows_removed, rows_removed_percent))
      
    return rem_df


    
 
    
    
    
