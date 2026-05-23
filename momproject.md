## Understanding U.S. Trends in Maternal Outcome Measurements (M.O.M.) - Why M.O.M.'s Matter

**Project description:** Increasing maternal mortality in the U.S. is an ongoing problem that health officials and researchers are eager to address. There are few studies that focus specifically on maternal mortality, and this project aims to drive that focus by identifying key areas of interest and proposing future work that could make prediction and prevention of maternal mortality a reality.

**Data Summary:** This project uses the dataset found here: [Maternal indicators in US States - Kaggle](https://www.kaggle.com/datasets/neharana404/maternal-indicators-in-us-states2016-2021), and the data was explored and analyzed using the sklearn and statsmodels libraries in Python. For this project, 33 numerical features were identified as potential contributors to maternal death and used in developing a generalized linear model (GLM), specifically Poisson regression, with Deaths/100,000 as the response, while the categorical and temporal features (State / Year) were used for data visualization and review.

### 1. Outlier Identification and Missing Value Handling

Outliers were identified via K-Means clustering of raw data as seen here:

<img src="images/clustering.jpg?raw=true"/> 

Optimal K function for plotting Silhouette, Calinski-Harabasz, and Davies-Bouldin scores: 

```python
# Libraries needed for function
import pandas as pd
import re
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load data from Kaggle (downloaded locally as CSV file) and drop raw counts/confidence interval columns as well as NA values
df = pd.read_csv(r'\MOMAnalysis\data\PRAMS_MCH_Indicators_2016_2021_Final.csv')
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

# Can then plot the scores as shown by accessing the df_scores dict

```
Clustering against state with optimal K revealed the following outliers:

<img src="images/outliers.png?raw=true"/>

Optimal K for K-Nearest-Neighbor imputation was also derived via the elbow method:

<img src="images/OptimalK_MOM_Analysis.png?raw=true"/>

```python 
# Libraries needed for deriving optimal K for KNN
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data from Kaggle (downloaded locally as CSV file) and drop raw counts/confidence interval columns
df = pd.read_csv(r'\MOMAnalysis\data\PRAMS_MCH_Indicators_2016_2021_Final.csv')
df = df[df.columns.drop(list(df.filter(regex='Confidence Interval|Unweighted')))]
df = df.rename(columns={'Site Name': 'State'})

# Create KNN friendly dataframe for finding optimal K (numerical data only, no 'State' data)
df_stateless = df.drop('State', axis=1)
df_knn = df_stateless.dropna()

# Split data into response (y) and features (X) then 30% test and 70% training data
X = df_knn.drop('Deaths/100000', axis=1)
y = df_knn['Deaths/100000']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Use elbow method to determine optimal K (optimal K discovered was 10, added line to visualize) by plotting error vs. K value
k_values = range(1, 21)
error_rates = []


for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error_rate = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error_rate)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(k_values, error_rates, marker='o')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Elbow Method for Finding Optimal K')
plt.axvline(x=10, linestyle='--', color='red', label="Optimal K (Elbow Point)")
plt.show()

# Impute NA values in numerical features (i.e., dataframe without 'State' data) with optimal K
X = df_stateless.drop('Deaths/100000', axis=1)
y = df_stateless['Deaths/100000']
imputer = KNNImputer(n_neighbors=10, weights="uniform")
X_imp = imputer.fit_transform(X)

# Remove NA response values and outliers determined from clustering
df_imp = pd.DataFrame(X_imp, columns=X.columns)
df_imp['State'] = df['State']
df_imp['Deaths/100000'] = y
df_final = df_imp.dropna()
df_final = df_final[~(df_final['Deaths/100000'] > 80)]
states = pd.DataFrame(df_final['State'])
```

### 2. Optimization of Poisson regression model via p-value and Bayesian Information Criterion (BIC) iterations

After cleaning the data, I scaled the data, split it into training and test datasets, and used the glm() method under statsmodels to fit the fully saturated model. I then checked the p-values for the coefficients in result.summary(). Given that many of the p-values were statistically insignificant (p > 0.05), I used an iterative process of removing the most insignificant terms until I arrived at a model where all terms were statistically significant. I also tracked BIC to assess whether the model improved as each term was removed.

```python
# Libraries needed for these functions
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


# Function for modeling maternal deaths as the response of maternal measurements
def poisson_model(x_data, y_data):
    # Initialize values for building model
    cols = list(x_data.columns)
    n = len(cols)
    x_val = {'Maternal_deaths': y_data}
    x_dict = {}
    formula = 'Maternal_deaths ~ '
    i_x = 1

    # Iterate over column names to add them as X1, X2, etc. variables and populate the model formula
    # Store the X# corresponding to each column in a dictionary
    for col in cols:
        if cols.index(col) < (n - 1):
            formula += f'X{i_x}' + ' + '
        else:
            formula += f'X{i_x}'
        x_val[f'X{i_x}'] = np.array(x_data[col])
        x_dict[f'X{i_x}'] = col
        i_x += 1
    
    # Create model and store results of the fitted model
    model = smf.glm(formula, data=x_val, family=sm.families.Poisson())
    result = model.fit()
    
    # Returns p-values for comparison of features, BIC score for model comparison, dict storing feature names,
    # results summary table, and the model itself
    return result.pvalues, result.bic, x_dict, result.summary(), model

# Initiate tracking of BIC, dropped terms, and model iteration number
dict_model_eval = {}
dropped_terms = []
model_i = 0

# Initialize while loop values using poisson_model function (saturated model)
result_summary, bic, x_ref, table, model_for_prediction = poisson_model(X_train_f, y_train_f)

# Drop intercept from result summary for easy comparison
no_int_p = result_summary.drop('Intercept')

# Iterate over columns to find best model based on p-values and BIC value
while len(result_summary) > 0 and max(no_int_p) > 0.05:
    no_int_p = result_summary.drop('Intercept')
    max_p_label = x_ref[no_int_p.idxmax()]

    if max(result_summary) > 0.05:
        dropped_terms += [max_p_label]
        X_train_f = X_train_f.drop(max_p_label, axis=1)
        X_test_f = X_test_f.drop(max_p_label, axis=1)
        result_summary, bic, x_ref, table, model_for_prediction = poisson_model(X_train_f, y_train_f)
        dict_model_eval['Model ' + f'{model_i}'] = int(bic)
    model_i += 1

# Save dropped terms and BIC value associated with each model for later reporting
model_data = {
    'Iteration': dict_model_eval.keys(),
    'Dropped Term': dropped_terms,
    'BIC': dict_model_eval.values()
}

# Create dataframe and print to view results
model_eval = pd.DataFrame(model_data)
print(model_eval.to_string())

```

The results of the iterations showed the following terms were dropped for lack of statistical significance:

<img src="images/dummy_thumbnail.jpg?raw=true"/>

To further check if the kept terms produced the optimal model, I intialized a new saturated model with the remaining significant terms and checked the BIC again for improvement upon removing a term. By iterating through all possible combinations of terms, I arrived at a new reduced model that could be compared to the new saturated model.

```python
# Initialize new saturated model with best model formula determined by p-value iteration
sat_model_form = 'Maternal_deaths ~ X1 + X2 + X3 + ' \
                 'X4 + X5 + X6 + X7 + X8 + X9 + ' \
                 'X10 + X11 + X12 + X13 + X14 + X15 + ' \
                 'X16 + X17 + X18 + X19 + X20 + X21 + X22 + X23'


# Create a function for generating list of possible new reduced formulas given a saturated formula
def gen_formula_list(first_form, x_ref_dict, x_data_init):
    formulas_to_eval = []
    pattern = r'\~(\s*(\S+))'
    match = re.search(pattern, first_form)
    term_1 = match.group(1).strip()
    x_data_lookup = {}

    for x_term in x_ref_dict:
        col_name = x_ref_dict[x_term]
        if x_term != term_1 and x_term != 'X2':
            str_to_replace = ' + ' + x_term
            formula_new = first_form.replace(str_to_replace, '')
        else:
            str_to_replace = x_term + ' + '
            formula_new = first_form.replace(str_to_replace, '')

        formulas_to_eval = formulas_to_eval + [formula_new]
        x_data_new = x_data_init.drop(col_name, axis=1)
        x_data_lookup[formula_new] = x_data_new

    return formulas_to_eval, x_data_lookup


# # Initialize BIC dictionary
BIC_results = {}


# Create a new saturated formula to start the process again
def sat_formula_create(model):
    sat_formula = 'Maternal_deaths ~ '
    n_index = len(model.index)
    i_index = 0
    for item in model.index:
        if item != 'Intercept' and i_index < n_index - 1:
            sat_formula = sat_formula + item + ' + '
        elif i_index == n_index - 1:
            sat_formula = sat_formula + item

    return sat_formula


# Define function for updating BIC dictionary
def update_bic_dict(x_data_ref, formulas_to_eval, bic_results_dict, best_bic=np.inf):
    best_model = None
    new_ssat_formula = ""

    for formula_test in formulas_to_eval:
        result_summary_test, bic_test, x_ref_test, table_test, model_for_prediction_test = poisson_model(
            x_data_ref[formula_test], y_train_f)
        if bic_test < best_bic:
            best_bic = bic_test
            best_model = result_summary_test
            new_sat_formula = sat_formula_create(result_summary_test)
            new_x_ref = x_ref_test
        bic_results_dict[formula_test] = bic_test

    return best_model, new_sat_formula, bic_results_dict, new_x_ref, best_bic


# Generate list of formulas to try
formulas_list, x_data_dict = gen_formula_list(sat_model_form, x_ref, X_train_f)

# Generate new model, formula, and BIC dictionary
new_model, new_formula, bic_dict, x_ref_new, bic_start = update_bic_dict(x_data_dict, formulas_list, BIC_results)

```

The best BIC was found when 5 additional features were removed from the model. However, because the BIC score improvement was <1 point compared to the previous BIC score, the newly saturated model with all 23 significant terms was determined to be the best possible model. The below table shows the optimal model and its coefficients.

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 3. Assessment and interpretation of saturated model

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 
