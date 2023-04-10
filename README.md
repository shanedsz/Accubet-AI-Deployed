# Accubet AI Deployed

## A data project that includes the entire Data Science lifecycle to predict the NBA games on March 27, 2023. Includes user interface run on Streamlit.

### 1. Data Scraping
The three Jupyter notebooks which include the "scraping", are taking the box scores straight for NBA.com, from the 2000-01 season all the way to the current NBA season. It saves each season to a single dataset per stat type (Scoring, Four Factors, and Advanced). It then exports each dataset to a CSV file where they will be ready for cleaning.

### 2. Data Cleaning
All three CSV files are brought into the Jupyter Notebook "data_cleaning_advanced.ipynb" and are cleaned using consistent methods. Keeping in mind the main goal is to use XGBRegression to predict winners, datasets are changed to only inform if the home team won or lost. All three datasets of different stat types are concatenated and made as rolling averages of the last 20 games each team played. The dataset for the March 27 games are saved into a seperate dataset and are saved into two csv files. 

### 3. XGBoost Model Configuration
The Jupyter Notebook "xgb_nba.ipynb" brings in the ready data and uses RandomizedSearchCV to choose the best parameters for the XGB model and runs the model with a 65% accuracy. It is then saved in JSON format so it can be used when deployed.

### 4.Model Deployment
The library Streamlit is used to make an interactive user interface where a user can choose from a dropdown menu which NBA game they would like predicted and the model set up in Step 3 predicts whether the home team will win or lose and returns the answer to the user. 
