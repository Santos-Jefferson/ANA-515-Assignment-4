#!/usr/bin/env python
# coding: utf-8

# # ANA 515 Assignment 4 - Data Analysis Project

# **Project selected:** Customer Segmentation using Machine Learning
# 
# **Link:** https://data-flair.training/blogs/r-data-science-project-customer-segmentation/
# 
# **Modeling:** Clustering (ML Unsupervised Learning)

# ### 1. Business Problem / Goal

# Marketing efforts/costs to advertize to customers without knowing if they will be interested in our products are really high.
# 
# The goal here is to identify potential customer groups to sell products. Once this is identified, the right product can be sold to the right customer, decreasing the marketing costs and increasing the profits.
# 

# ### 2. Dataset retrieval

# This dataset was retrived from Kaggle website in a .csv format.
# 
# **Kaggle link:** https://www.kaggle.com/shwetabh123/mall-customers

# ### 3. Importing and saving the dataset using Pandas in Python
# The file Mall_Customers.csv is saved in the same folder of this jupiter notebook.

# In[29]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
sns.set_style("darkgrid")
import scipy


# In[30]:


# Using pandas function called read_csv and passing the filename as an argument
data = pd.read_csv('Mall_Customers.csv')


# In[31]:


# Sample of the data (start and end of the dataset)
data


# ### 4. Describing the Data

# In[32]:


# Data #rows and #columns
rows = data.shape[0]
cols = data.shape[1]
print(f"This dataset has {rows} rows and {cols} columns in total.")


# In[33]:


# Variable Names and their types
print(f"This dataset has the following column name and types:")
data.dtypes


# In[34]:


# Dataset summary (means, sd, min/max)
print(f"This dataset has the following summary statistics:")
data.describe()


# In[35]:


# Dataset NAs
print(f"This dataset has the following NAs values per column:")
data.isnull().sum()


# In[36]:


# Info for categorical variables
print(f"This dataset has the following stats for the Categorical column:")
data.describe(include=object)


# ### 5. Data Preparation, Missing Values and Errors
# As we can see above on #Dataset NAs section, the sum of null values for each column is 0, meaning we don't have Null Values in this dataset.
# 
# When we talk about raw data and the common issues, we can include:
# 1. Poor data like noise data - When you have different databases with no info about each column, you end up having poor data.
# 2. Dirty data - duplicated ata, errors, hardware/network inconsistencies.
# 3. Missing values - pieces of the data may be missing from the collection process.
# 4. Wrong data size - unbalanced datasets with different sizes of columns and rows from different datasources.
# 5. Misrepresentation of the population data - input from different databases, lacking what is available or not.
# 
# Talking about data preparation, I will drop CustomerID column from the dataset, as it will be irrelevant for my model on the next section. This CustomerID is only a index number from 0 to 199 sequentialy.

# In[37]:


# Dropping CustomerID as it won't be usefull for the model and graphs below
data.drop('CustomerID', axis=1, inplace=True)


# In[38]:


# Renaming columns to lowercase and removing spaces
data.columns = ['genre', 'age', 'annual_income_k$', 'spending_score_1_100']


# In[39]:


# Converting categoric columns to numeric
print(f"This dataset has the following data types:")
data.genre = pd.Categorical(data.genre)
data.dtypes


# In[40]:


# Converting Gender column from Male/Female to 0 or 1 to be used in the model too.
data.genre = data.genre.cat.codes
data


# In[41]:


# Checking the distribution of genre after converted to 0 or 1
print(f"This dataset has the following counts for Genre, mapped 1 to Male, 0 to Female:")
data.genre.value_counts()


# ### 6. The Model

# We are going to use K-means clustering algorithm on this data.
# 
# First thing, we check the Elbow method to identify the best number of clusters to use in our clustering process.

# In[42]:


# Elbow method
distortions = []
n_clusters = range(1, 10)
for num in n_clusters:
    kmean = KMeans(n_clusters=num)
    kmean.fit(data)
    distortions.append(kmean.inertia_)


# I would say that from 4 to 6 number of clusters we should have our best scores in this process.

# In[43]:


# Plot the elbow
plt.figure(figsize=(16,8))
plt.plot(n_clusters, distortions, 'bx-')
plt.xlabel('Num')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal N of Clusters')
plt.show()


# In[44]:


range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
X = data.drop('genre', axis=1)

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X.iloc[:, 0], X.iloc[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()


# The best silhouette score was for 6 clusters, as we could see above in the charts and results:
# 
# For n_clusters = 6 The average silhouette_score is : **0.45**
# 
# Now, let's run again using n_clusters = 6 for the rest of our analysis.

# In[45]:


# Running the algo again with 6 as number of clusters
kmean = KMeans(n_clusters=6)
kmean.fit(data)
# Getting the labels (clusters numbers to a variable)
clusters = kmean.labels_


# ### 7. The Outcome
# Let's make some analysis of the results now. 
# 
# First, let's copy the original dataframe and add a column called "Clusters" from the algo results.
# 
# Second, let's replace the 0 and 1 for genre and insert again the words "Male" and "Female"

# In[46]:


# Copying data, creating columns and replacing number for strings
new_data = data.copy()
new_data['clusters'] = clusters
new_data['genre'] = new_data['genre'].replace(0, "Female")
new_data['genre'] = new_data['genre'].replace(1, "Male")
new_data


# 
# 
# 
# 
# 
# Creating bloxplots and bar charts for a better visualization of each cluster and its caracteristics (Age, Income, Spending Score)

# In[47]:


numeric_columns = ['age', 'annual_income_k$', 'spending_score_1_100']
for i in numeric_columns:
    plt.figure(figsize=(6,4))
    ax = sns.boxplot(x = 'clusters', y=i, data=new_data)
    plt.title('\nBoxplot {}\n'.format(i), fontsize=15)


# In[48]:


categ_column = ['genre']

for i in categ_column:
    plt.figure(figsize=(11,7))
    ax = sns.countplot(data = new_data, x = 'clusters', hue = i)
    plt.title(f"Count Plot {i}", fontsize=10)
    ax.legend(loc="upper center")
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center',
                   xytext = (0, 10),
                    textcoords = 'offset points')
    sns.despine(right=True,top = True, left = True)
    ax.axes.yaxis.set_visible(False)
    plt.show()


# Creating a 3D scatter plot to find the clusters location by color

# In[49]:


fig = plt.figure(figsize=(30,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(new_data.age[new_data.clusters == 0], new_data["annual_income_k$"][new_data.clusters == 0], new_data["spending_score_1_100"][new_data.clusters == 0], c='blue', s=60)
ax.scatter(new_data.age[new_data.clusters == 1], new_data["annual_income_k$"][new_data.clusters == 1], new_data["spending_score_1_100"][new_data.clusters == 1], c='red', s=60)
ax.scatter(new_data.age[new_data.clusters == 2], new_data["annual_income_k$"][new_data.clusters == 2], new_data["spending_score_1_100"][new_data.clusters == 2], c='green', s=60)
ax.scatter(new_data.age[new_data.clusters == 3], new_data["annual_income_k$"][new_data.clusters == 3], new_data["spending_score_1_100"][new_data.clusters == 3], c='orange', s=60)
ax.scatter(new_data.age[new_data.clusters == 4], new_data["annual_income_k$"][new_data.clusters == 4], new_data["spending_score_1_100"][new_data.clusters == 4], c='purple', s=60)
ax.scatter(new_data.age[new_data.clusters == 5], new_data["annual_income_k$"][new_data.clusters == 5], new_data["spending_score_1_100"][new_data.clusters == 5], c='black', s=60)
ax.view_init(30, 185)
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
ax.set_zlabel('Spending Score (1-100)')
plt.show()


# ### 8. Visualization
# 
# Below, I decided to plot some exploratory analysis before modeling the data with K-Means Algo.
# Usually, I would put these info before modeling section.

# In[50]:


# Bar graph view
data.genre.value_counts().plot(kind="bar",
                             title="Gender Counts",
                             xlabel="Gender",
                             ylabel="Count",
                             rot=0);


# In[51]:


# Pie graph view
data.genre.value_counts().plot(kind="pie",
                             title="Gender Counts");


# In[52]:


# Boxplot graph view
data.plot(y="age",
          kind="box",
          title="Boxplot - Age");


# In[53]:


# Boxplot graph view
data.plot(y="annual_income_k$",
          kind="box",
          title="Boxplot - Annual Income");


# In[54]:


# Boxplot graph view
data.plot(y="spending_score_1_100",
          kind="box",
          title="Boxplot - Spending Score");


# In[55]:


# Scatter Matrix and Distribuition to compare all the columns against each other
pd.plotting.scatter_matrix(data, diagonal="kde", figsize=(15,15));


# In[56]:


# Heat map for correlation
fig = plt.figure(figsize=(8, 7))
sns.heatmap(data.corr(), annot=True);


# ### 9. Results and  Model Evaluation
# This section I would like to create some kind of description for each cluster created during the Modeling section.
# 
# **Cluster 0:** General Public
#     
#     1. Average annual income of U$ 25,000.
#     2. Age range of about 20–68 years, average of 45 years.
#     3. Women dominate.
#     4. Low spending_score (5–40).
# 
# **Cluster 1:** Elderly General Public.
# 
#     1. Average annual income of U$ 55,000.
#     2. Age range of about 45–70 years, average of 55 years.
#     3. Women dominate.
#     4. Moderate spending_score (30–60).
# 
# **Cluster 2:** Students
# 
#     1. Average annual income of U$ 25,000.
#     2. Age range of about 20–35 years, average of 22 years.
#     3. Women dominate.
#     4. High spending_score (60–98).
# 
# **Cluster 3:** Entrepreneurs
# 
#     1. Average annual income of U$ 80,000.
#     2. Age range of about 20–55 years, average of 43 years.
#     3. Males predominate.
#     4. Low spending_score (5–40).
# 
# **Cluster 4:** Young Entrepreneurs
# 
#     1. Average annual income of U$ 60,000.
#     2. Age range of about 20–55 years, average of 43 years.
#     3. Women dominate.
#     4. Moderate spending_score (30–60).
# 
# **Cluster 5:** Young Entrepreneurs II
# 
#     1. Average annual income of U$ 75,000.
#     2. Age range of about 25–37 years, average of 31 years.
#     3. Women dominate.
#     4. High spending_score (65–90).
