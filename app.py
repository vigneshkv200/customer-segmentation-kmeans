import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

data=pd.read_csv("Mall_Customers.csv")

X= data[["Annual Income (k$)","Spending Score (1-100)"]]

scaler=joblib.load("scalermodel.pkl")
kmeans=joblib.load("kmeansmodel.pkl")


Scaled=scaler.transform(X)
data["Cluster"]= kmeans.predict(Scaled)

st.title("Customer Segmentation System")

income=st.number_input("Annual income (k$)")
spending=st.number_input("Spendings Score")

if st.button("predict"):
    scaled=scaler.transform([[income,spending]])
    cluster=kmeans.predict(scaled)[0]

    if cluster==0:
        result = "Premium Customer"
    elif cluster== 1:
        result = "Budget Customer"
    elif cluster== 2:
        result = "Regular Customer"
    elif cluster== 3:
        result = "Potential Customer"
    else:
        result ="Impulsive Buyer"

    st.success(result)
    print(income,spending,cluster)
    
    fig,ax= plt.subplots()
    
    ax.scatter(
        data["Annual Income (k$)"],
        data["Spending Score (1-100)"],
        c=data["Cluster"],
        cmap="viridis",
        alpha = 0.6
    )
    
    ax.scatter(
        income,spending,color="red",s=200,label="New Customer")
    
    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")
    
    st.pyplot(fig)
            