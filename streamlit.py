import pandas as pd
import streamlit as st
import pickle


# Page Layout
st.set_page_config(page_title='DataMinds', layout='wide')


# Two pages
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])


# Home page
if app_mode == "Home":
    st.title("Credit Risk Analysis")
    st.header("Building a Predictive Model for Loan Default Status")
    st.markdown("*By DataMinds*")
    st.image("loan.jpg")

    #Intro
    st.write("Asessing credit risk is crucial for lenders and financial institutions in today's finacial landscape. This project aims to build predictive model to determine the likelihood of loan default as well as generating insights relevant to financial institutions.")
    
    # Display train data
    st.write("Train Data Sample:")
    loan_df = pd.read_csv("loan_df.csv")
    st.write(loan_df.head())

# Function to encode categorical features
def encoder(df):
    encoded_data = pd.get_dummies(data=df, columns=['home', 'intent', 'age_group'], drop_first=True)
    
    required_features = [
        'income', 'emp_length', 'amount', 'rate', 'percent_income', 'cred_length',
        # Add all possible one-hot encoded columns for 'home', 'intent', and 'age_group'
        # For example, if the user only selects "Rent" in the input, pd.get_dummies will only create the home_Rent column, omitting home_Own and home_Mortgage, which causes the shape mismatch with the model.
        'home_Other', 'home_Own', 'home_Rent', 'intent_Education', 'intent_Homeimprovement', 'intent_Medical', 'intent_Personal', 'intent_Venture',
        'age_group_25-29', 'age_group_30-34', 'age_group_35-39', 'age_group_40-44', 'age_group_45-49', 'age_group_50-54', 'age_group_55-59',
        'age_group_60-64', 'age_group_65-69', 'age_group_70-74', 'age_group_75-79', 'age_group_80-84', 'age_group_85-89', 'age_group_90-94',
        'age_group_95-99'
    ]
    
    # Add any missing columns with zeros
    for col in required_features:
        if col not in encoded_data:
            encoded_data[col] = 0
    
    # Reorder columns to match the order the model expects
    encoded_data = encoded_data[required_features]
    return encoded_data
    

# Prediction page
if app_mode == "Prediction":
    # Input features
    st.sidebar.header("Enter Input Parameters")
    
    age_group = st.sidebar.selectbox("Age Group", ['20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', 
                                                   '70-74', '75-79', '80-84', '85-89', '90-94', '95-99'])
    income = st.sidebar.number_input("Annual Income (in $)", 3000, 2500000, 90000)
    home = st.sidebar.selectbox("Home Ownership", ["Mortgage", "Rent", "Own", "Other"])
    emp_length = st.sidebar.slider("Employment Length (years)", 0, 50, 7)
    intent = st.sidebar.selectbox("Loan Intent", ["Debtconsolidation", "Education", "Medical", "Venture", "Personal", "Homeimprovement"])
    amount = st.sidebar.number_input("Loan Amount (in $)", 500, 35000, 10000)
    rate = st.sidebar.number_input("Interest Rate (%)", 5.0, 25.0, 5.0)
    cred_length = st.sidebar.slider("Credit History Length (years)", 0, 30, 15)
    percent_income = amount/income


    # Creating a dictionary out of input parameters
    data_dict = {
        'income': [income],
        'emp_length': [emp_length],
        'amount': [amount],
        'rate': [rate],
        'percent_income': [round(percent_income, 2)],
        'cred_length': [cred_length],
        'home': [home],
        'intent': [intent],
        'age_group': [age_group],
    }


    # Converting the dictionary to a pandas dataframe
    input_df = pd.DataFrame(data_dict)


    # Display input data
    st.subheader("Input Parameters:")
    st.write(input_df)
    st.write("Shape of input data:", input_df.shape)

    
    # Encoding categorical features
    input_data = encoder(input_df)

    if st.button("Predict"):
            
        # Loading a prebuilt and saved model
        model = pickle.load(open('pipeline.pkl', 'rb'))


        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]


        # Debugging outputs
        st.write("Prediction:", prediction)
        st.write("Prediction probability:", prediction_proba)

                 
        # Display results
        st.subheader("Prediction")
        if prediction[0] == 'Y':
            st.write("The model predicts: Default")
        else:
            st.write("The model predicts: No Default")

        st.subheader("Prediction Probability")
        st.write(f"Probability of Default: {prediction_proba[0]:.2f}")