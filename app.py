import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


st.set_page_config(layout="wide")

# Simple Text
st.header('Streamlit Dashboard')
st.write('This is streamlit dashboard')

df = pd.read_csv('https://raw.githubusercontent.com/deandrasann/house_dataset/refs/heads/master/CC%20GENERAL.csv', delimiter=',')

# Preprocessing
df_new = df.drop('CUST_ID', axis = 1)

df_new['MINIMUM_PAYMENTS'].fillna(df_new['MINIMUM_PAYMENTS'].median(), inplace=True)
df_new['CREDIT_LIMIT'].fillna(df_new['CREDIT_LIMIT'].median(), inplace=True)

X = df_new.astype(float).values
scaler = StandardScaler().fit(X)
X_new= scaler.transform(X)

def run_kmeans(df_new, X_new):
    st.header('KMeans')
    k_means = KMeans(n_clusters = 3, random_state = 42)
    k_means.fit(X_new)
    labels = k_means.labels_
    df_new['cluster_labels'] = labels
    st.write('Visualization with Matplotlib')
    st.markdown("---")
    x1 = df_new['PURCHASES']
    x2 = df_new['PAYMENTS']
    fig_matplotlib = plt.figure(figsize=(8,6))
    u_labels = np.unique(labels) 
    for i in u_labels:
        plt.scatter(x1[df_new['cluster_labels'] == i] , 
                    x2[df_new['cluster_labels'] == i] , label = i)
    plt.scatter(x1,x2, c=k_means.labels_, cmap='rainbow')
    plt.xlabel(x1.name,  fontsize=20)
    plt.ylabel(x2.name,  fontsize=20)
    plt.title('K-means clustering',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    st.pyplot(fig_matplotlib)
    st.write('Visualization with Seaborn')
    st.markdown("---")
    fig_seaborn = plt.figure(figsize=(8,6))
    x_val = 'PURCHASES'
    y_val = 'PAYMENTS'
    sns.scatterplot(x=x_val, y=y_val, hue='cluster_labels', 
                    data=df_new, palette='Paired')
    plt.legend(loc='lower right')
    st.pyplot(fig_seaborn)
    st.write('Visualization with Plotly Express')
    st.markdown("---")
    x_val = 'PURCHASES'
    y_val = 'PAYMENTS'
    z_val = 'BALANCE'
    fig_plotly = px.scatter_3d(df_new, x=x_val, y=y_val, z=z_val, 
                               color='cluster_labels', labels='cluster_labels')
    st.plotly_chart(fig_plotly)


def run_kmedoids(df_new, X_new):
    st.header('KMedoids')
    k_medoids = KMedoids(n_clusters = 4, random_state = 42)
    k_medoids.fit(X_new)
    labels = k_medoids.labels_
    df_new['cluster_labels'] = labels
    df_new.head()
    st.write('Visualization with Matplotlib')
    st.markdown("---")
    x1 = df_new['PURCHASES']
    x2 = df_new['PAYMENTS']
    fig_matplotlib = plt.figure(figsize=(8,6))
    u_labels = np.unique(labels) 
    for i in u_labels:
        plt.scatter(x1[df_new['cluster_labels'] == i] , 
                    x2[df_new['cluster_labels'] == i] , label = i)
    plt.scatter(x1,x2, c=k_medoids.labels_, cmap='rainbow')
    plt.xlabel(x1.name,  fontsize=20)
    plt.ylabel(x2.name,  fontsize=20)
    plt.title('K-medoids clustering',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    st.pyplot(fig_matplotlib)
    st.write('Visualization with Seaborn')
    st.markdown("---")
    fig_seaborn = plt.figure(figsize=(8,6))
    x_val = 'PURCHASES'
    y_val = 'PAYMENTS'
    sns.scatterplot(x=x_val, y=y_val, hue='cluster_labels', 
                    data=df_new, palette='Paired')
    plt.legend(loc='lower right')
    st.pyplot(fig_seaborn)
    st.write('Visualization with Plotly')
    st.markdown("---")
    x_val = 'PURCHASES'
    y_val = 'PAYMENTS'
    z_val = 'BALANCE'
    fig_plotly = px.scatter_3d(df_new, x=x_val, y=y_val, z=z_val, 
                               color='cluster_labels', labels='cluster_labels')
    st.plotly_chart(fig_plotly)


col1, col2 = st.columns(2)
with col1:
    run_kmeans(df_new, X_new)
with col2:
    run_kmedoids(df_new, X_new)






  
  


