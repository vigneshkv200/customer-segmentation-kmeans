
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("Mall_Customers.csv")

print(data.head())

X=data[["Annual Income (k$)","Spending Score (1-100)"]]

scaler =StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss =[]

for i in range(1,20):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,20),wcss,marker ="o")

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


kmeans =KMeans(n_clusters=5,random_state=42)
clusters = kmeans.fit_predict(X_scaled)


data["Cluster"] =clusters

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=data["Annual Income (k$)"],
    y=data["Spending Score (1-100)"],
    hue=data["Cluster"],
    palette="viridis"
)

plt.title("Customer Segmentation")
plt.show()

print(data.head())
