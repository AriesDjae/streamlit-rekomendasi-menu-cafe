import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori

# load dataset

df = pd.read_csv("bread basket.csv")
df['date_time'] = pd.to_datetime(df['date_time'], format="%d-%m-%Y %H:%M")

df["month"] = df['date_time'].dt.month
df["day"] = df['date_time'].dt.weekday
# df["hour"] = df['date_time'].dt.hour

df["month"].replace([i for i in range(1, 12+1)], ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], inplace=True)
df["day"].replace([i for i in range(6+1)], ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], inplace=True)

st.title("Market Basket Analysis Menggunakan Algoritma Apriori")

def get_data(period_day = '', weekday_weekend= '', month= '', day = ''):
    data = df.copy()
    filtered = data.loc[
        (data["period_day"].str.contains(period_day)) & 
        (data["weekday_weekend"].str.contains(weekday_weekend)) &
        (data["month"].str.contains(month.title())) &
        (data["day"].str.contains(day.title()))
    ]
    return filtered if filtered.shape[0] else "No Result!"

def user_input_feutures():
    item = st.selectbox("Item", ['Bread','Scandinavian','Hot chocolate','Jam','Cookies','Muffin','Coffee','Pastry','Medialuna','Tea','Tartine','Basket','Mineral water','Farm House','Fudge','Juice',"Ella's Kitchen Pouches",'Victorian Sponge','Frittata','Hearty & Seasonal','Mighty Protein','Chicken sand','Coke','My-5 Fruit Shoot','Pick and Mix Bowls','Smoothies','Cake','Soup'])
    period_day = st.selectbox('Period Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    weekday_weekend = st.selectbox('Weekeday / Weekend', ['Weekend', 'Weekday'])
    month = st.select_slider('Month', ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    day = st.select_slider('Day', ["Mon", "Tue", "Wed","Thu", "Fri", "Sat", "Sun"], value="Sat")

    return period_day, weekday_weekend, month, day, item

period_day, weekday_weekend, month, day, item = user_input_feutures()

data = get_data(period_day.lower(),weekday_weekend.lower(), month, day)

def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    
if type(data) != type ("No Result!"):
    item_count = data.groupby(["Transaction", "Item"])["Item"].count().reset_index(name="Count")
    item_connt_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    item_connt_pivot = item_connt_pivot.applymap(encode)

    support = 0.01
    frequent_items = apriori(item_connt_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(frequent_items, metric=metric, min_threshold=min_threshold)[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values('confidence', ascending=False, inplace=True)

def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ", ".join(x)
    
def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    return list(data.loc[data["antecedents"] == item_antecedents].iloc[0,:])

if type(data) != type("No Result"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(f" Jika konsumen membeli **{item}**, maka membeli **{return_item_df(item)[1]}** secara bersamaan")