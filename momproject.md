## Understanding U.S. Trends in Maternal Outcome Measurements (M.O.M.) - Why M.O.M.'s Matter

**Project description:** Increasing maternal mortality in the U.S. is an ongoing problem that health officials and researchers are eager to address. There are few studies that focus specifically on maternal mortality, and this project aims to drive that focus by identifying key areas of interest and proposing future work that could make prediction and prevention of maternal mortality a reality.

**Data Summary:** This project uses the dataset found here: [Maternal indicators in US States - Kaggle](https://www.kaggle.com/datasets/neharana404/maternal-indicators-in-us-states2016-2021), and the data was explored and analyzed using the sklearn and statsmodels libraries in Python. This dataset consists of 167 columns, which include maternal deaths in the form of counts/100,000, year of measurement, US state, and various maternal indicators such as BMI, prenatal usage, insurance coverage, and more. There are 226 rows, and upon routine exploration, many missing values that were addressed via K-Nearest-Neighbor imputation where appropriate or removed in the case of the response (Deaths/100,000). For this project, 33 numerical features were identified as potential contributors to maternal death and used in developing a generalized linear poisson regression with Deaths/100,000 as the response, while the categorical and temporal features (State / Year) were used for data visualization and review.

### 1. Outlier Identification and Missing Value Handling

Outliers were identified via K-Means clustering of raw data as seen here:

<img src="images/dummy_thumbnail.jpg?raw=true"/> 

With an optimal K derived from visual assessment of sklearn metrics: 

```Python
# Libraries needed
import pandas as pd
import re
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Loaded data
df = pd.read_csv(r'C:\Users\dinos\PycharmProjects\MOMAnalysis\data\PRAMS_MCH_Indicators_2016_2021_Final.csv')
df = df[df.columns.drop(list(df.filter(regex='Confidence Interval|Unweighted')))]
df = df.rename(columns={'Site Name': 'State'})
df = df.dropna()

# Figure out optimal K for K-means clustering of MOM data
def find_optimal_k(data):
    seed_random = 5
    fitted_kmeans = {}
    labels_kmeans = {}
    df_scores = []

    # Guess optimal value is less than half the number of data points (138)
    k_values_to_try = np.arange(2, 78)
    for n_clusters in k_values_to_try:
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=seed_random)
        labels_clusters = kmeans.fit_predict(data)

        # Store fitted model and cluster labels in dictionaries
        fitted_kmeans[n_clusters] = kmeans
        labels_kmeans[n_clusters] = labels_clusters

        # Calculate scores and save for reference
        silhouette = silhouette_score(data, labels_clusters)
        ch = calinski_harabasz_score(data, labels_clusters)
        db = davies_bouldin_score(data, labels_clusters)
        tmp_scores = {"n_clusters": n_clusters,
                      "silhouette_score": silhouette,
                      "calinski_harabasz_score": ch,
                      "davies_bouldin_score": db}
        df_scores.append(tmp_scores)

    # Dataframe of clustering scores for plotting, using n_clusters as index
    df_scores = pd.DataFrame(df_scores)
    df_scores.set_index("n_clusters", inplace=True)

    return df_scores, labels_kmeans, fitted_kmeans

# Plot k-optimization metrics for Kmeans
info = find_optimal_k(df)
scores = info[0]
scores['silhouette_score'].plot()
plt.title('Silhouette Score vs. K')
plt.show()
scores['calinski_harabasz_score'].plot()
plt.title('Calinski-Harabasz Score vs. K')
plt.show()
scores['davies_bouldin_score'].plot()
plt.title('Davies-Bouldin Score vs. K')
plt.show()

```

<img src="images/dummy_thumbnail.jpg?raw=true"/>

Optimal K for K-Nearest-Neighbor imputation was also derived via the elbow method:

```Python
# Libraries needed
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Loaded data
df = pd.read_csv(r'C:\Users\dinos\PycharmProjects\MOMAnalysis\data\PRAMS_MCH_Indicators_2016_2021_Final.csv')
df = df[df.columns.drop(list(df.filter(regex='Confidence Interval|Unweighted')))]
df = df.rename(columns={'Site Name': 'State'})

# Create KNN friendly df for finding optimal k (numerical data only)
df_stateless = df.drop('State', axis=1)
df_knn = df_stateless.dropna()

# Use elbow method to determine optimal k (optimal k discovered was 10, added line to visualize)
X = df_knn.drop('Deaths/100000', axis=1)
y = df_knn['Deaths/100000']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

k_values = range(1, 21)
error_rates = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rate = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error_rate)

plt.figure(figsize=(8, 6))
plt.plot(k_values, error_rates, marker='o')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Elbow Method for Finding Optimal K')
plt.axvline(x=10, linestyle='--', color='red', label="Optimal K (Elbow Point)")
plt.show()
```

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 1. Statistical significance of maternal indicators

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

```javascript
if (isAwesome){
  return true
}
```

### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 
