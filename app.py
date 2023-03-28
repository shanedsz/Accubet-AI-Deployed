import xgboost as xgb
import streamlit as st
import pandas as pd

# Loading XGB model
model = xgb.XGBClassifier()
model.load_model('xgb_nba.json') 

#loading original dataframe
df_todays_games = pd.read_csv('todays_schedule.csv',parse_dates = ['GAME_DATE'], infer_datetime_format= True)
# Changing date and season column to an integer
df_todays_games['GAME_DATE'] = df_todays_games['GAME_DATE'].apply(lambda x: x.toordinal())
todays_games = df_todays_games['GAME_ID'].to_list()
df_todays_games = df_todays_games.set_index('GAME_ID')
df_todays_games = df_todays_games.drop(columns=['SEASON_YEAR','HOME_WL'])

#Caching the model for faster loading
@st.cache

def predict (game_id):
   prediction =  model.predict(df_todays_games.loc[[game_id]])
   return prediction


st.title('NBA Game Winner Predictor')
st.header('Choose which game youwould like predicted:')
game_id = st.selectbox('Game:', todays_games)

if st.button('Predict Winner'):
    winner = predict(game_id)
    if winner == 0:
        st.success(f'The predicted winner of this game is the home team')
    elif winner == 1:
        st.success(f'The predicted winner of this game is the away team')

