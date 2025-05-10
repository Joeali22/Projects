!pip install numpy==1.23.5
!pip install scikit-learn==1.2.2
!pip install scikit-learn-extra==0.3.0 
from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#extracting the file from google drive
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/dataset/train.csv'
df = pd.read_csv(file_path)

#taking a sample from the file to reduce memory consuming    
df = df.sample(n=10000, random_state=42)

#data cleaning
df.drop_duplicates(inplace=True)      
df.dropna(inplace=True)            
df['Dates'] = pd.to_datetime(df['Dates'])  

#extracting the details of time
df['year'] = df['Dates'].dt.year
df['month'] = df['Dates'].dt.month
df['day'] = df['Dates'].dt.day
df['hour'] = df['Dates'].dt.hour
df['DayOfWeek']=df['Dates'].dt.dayofweek
# filter the Coordinates to be in san fransesco
df = df[(df['X'] > -123) & (df['X'] < -121)]
df = df[(df['Y'] > 37) & (df['Y'] < 38)]

#choose the columns 
features = ['X', 'Y', 'hour', 'month']
X = df[features]

#Normalization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# compute the best k by silhoutte score
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmedoids = KMedoids(n_clusters=k, random_state=42)
    labels = kmedoids.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

#draw the best k using silhoutte score
plt.figure(figsize=(8, 4))
plt.plot(K_range, silhouette_scores, marker='o')
plt.title('Silhouette Score vs K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# apply k-medoids algorithm with best k
best_k = K_range[silhouette_scores.index(max(silhouette_scores))]
print(f" best number of medoids (k) is: {best_k}")

kmedoids = KMedoids(n_clusters=best_k, random_state=42)
final_labels = kmedoids.fit_predict(X_scaled)
df['cluster'] = final_labels

#draw the final result using x,y
plt.figure(figsize=(8, 6))
plt.scatter(df['X'], df['Y'], c=final_labels, cmap='viridis', s=10)
plt.title(f'K-Medoids Clustering (K={best_k})')
plt.xlabel('Longitude (X)')
plt.ylabel('Latitude (Y)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

day_counts = df['DayOfWeek'].value_counts().sort_index()
  
day_names = ['Monday', 'Tuesday', 'Wedensday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

plt.figure(figsize=(8, 6))
plt.pie(day_counts, labels=day_names, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of crimes by Day of the Week')
plt.axis('equal')  
plt.show()
