import streamlit as st
from ingest import analyze_predicted_price, data_file
import pandas as pd
import numpy as np
import matplotlib

#################################################################
################## Page Settings ################################
#################################################################
st.set_page_config(page_title="My Streamlit App", layout="wide")
st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)

#################################################################
################## Page Header ##################################
#################################################################
st.header("Predicting Price For Used Cars")
st.write("My Application uses existing car specification data to predict whether this car is going to be high price or low price as car prices' flucuate nowadays. This application can help consumers to select cars based on specifications. The plots car help consumers to better visualize specifications that they care about to make decisions when buying a car.")
st.markdown('---')

################## Sidebar Menu #################################
page_selected = st.sidebar.radio("Menu", ["Home", "Model"])

################################################################
################## Home Page ###################################
################################################################
if page_selected == "Home":
    
    ######### Load labeled data from datastore #################

    df = pd.read_csv(data_file)
    df['color'] = df['price'].apply(lambda x: 'orange' if x == 0 else 'skyblue')
    df['price_range'] = df['price'].apply(lambda x: 'high' if x == 1 else 'low')
    

    ######### Fuel Type Filter ################################
    fueltype = st.sidebar.selectbox("FuelType", ['All', 'Petrol', 'Diesel', 'Hybrid'])
    
    ######### Brand Type Filter ################################
    brand = st.sidebar.selectbox("Brand", ['All', 'merc', 'ford', 'vw', 'bmw', 'hyundi', 'toyota', 'skoda', 'audi', 'vauxhall'])

    ######### Apply filters ####################################
    if fueltype != "All":
        df = df.loc[df.fuelType == fueltype, :]
    if brand != "All":
        df = df.loc[df.brand == brand, :]
    
    ######### Main Story Plot ###################################
    col1, col2 = st.columns((2,1))
    with col1: 
        ax = pd.crosstab(df.fuelType, df.price_range).plot(
                kind="bar", 
                figsize=(6,2), 
                xlabel = "Price Range",
                color={'low':'orange', 'high': 'skyblue'})
        st.pyplot(ax.figure)
    with col2:
        st.write('This plot shows the total counts of high and low price for each fueltype. If you are planning to buy a cheaper car, you might want to take closer look to petrol cars.')
    st.markdown('---')

    ######### Brand vs Price Plot ###################################
    col1, col2 = st.columns((2,1))
    with col1: 
        ax = pd.crosstab(df.brand, df.price_range).plot(
                kind="bar", 
                figsize=(6,2), 
                xlabel = "Price Range",
                color={'low':'orange', 'high': 'skyblue'})
        st.pyplot(ax.figure)
    with col2:
        st.write("This plot shows the total counts of high and low price for each brand. You can also look at the combination of fueltype and brand's price.")
    st.markdown('---')
    
    title = st.text_input('Year (Enter Up to 2020)', 2012)
    st.write("The year you've entered is", title)
    
    if title != "All":
        df = df.loc[df.year == int(title), :]

    ######### Year vs Price Plot ###################################
    col1, col2 = st.columns((2,1))
    with col1: 
        ax = pd.crosstab(df.year, df.price_range).plot(
                kind="bar", 
                figsize=(6,2), 
                xlabel = "Price Range",
                color={'low':'orange', 'high': 'skyblue'})
        st.pyplot(ax.figure)
    with col2:
        st.write("This plot shows the total counts of high and low price for each year that user enteres. Enter a year up to 2020 to see whether it's more likely to be high price or low price. Feel free to select fueltype and car brand as well to see how the combination affects the car price.")
    st.markdown('---')


################################################################
################ Model Page ######################
################################################################
else:
    st.subheader("Data Exploration")
    col1, col2 = st.columns((1,40))
    with col2:
        st.write("The data was found on Kaggle for one dataset. The dataset was randomly splitted into a training and testing dataset. I used training dataset for model training and pipeline building and I used testing dataset for this application. Vehicle price has been increasing for the past two years. My goal was to build a machine learning pipeline that could help me to accurately predict the vehicle price. I've included three major factors a person would consider while purchasing a car in the home page. You can make any combination of those three factors and see from the graph wheter you have a higher chance of buying a cheap car or an expensive car.")
    st.subheader('Model Buidling and Evaluation')
    col1, col2 = st.columns((1,40))
    with col2:
        st.write("After exploration of the dataset, some features seemed unnecessary  to be kept thus I dropped some of them. I dropped feature tax, mpg and engine size because tax isn't correlated with car itself. Mpg and engine size would be highly correlated with model since certain car model is determined to be high-emission type of car. I changed output variable 'price' into a two-level categorical variable for two main reasons. Firstly, Categorical variables would be more efficient for me to build multiple models. Secondly, a statistical report indicates that $20,000 can usually be a price to differentiate luxury and normal cars for used cars. I built a logistic regression at first and the model performed well. It had a 92% training accuracy and 92.7% testing accuracy. I built a Knn model for the second pipeline with hyperparameter tunning using gridsearch. It performed even better than the first model. It had a 92% training accuracy and 93% testing accuracy. I’m satisfied with the second model’s performance. I found out that the second model used ‘n_neighbors’ = 6 as parameter, thus I rebuilt the second model with n_neighbors = 6 in the final pipeline and saved it in my local computer.")