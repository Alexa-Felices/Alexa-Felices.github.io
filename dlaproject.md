## Building a Deep Learning Algorithm to Predict the Probability a News Story is About Health and Wellness

**Problem Statement:**
A hypothetical client wishes to target users reading about health and wellness for their products, as they are more likely to consider and purchase them. In order to accurately predict what a news story may contain, the model will be trained on a dataset containing 200k news headlines from the year 2012 to 2018 obtained from HuffPost, utilizing the "Healthy Living" category of articles to determine which articles are considered "health and wellness."

### 1. Imports, Mounting Drive, and Setting Runtime
This model uses k-train as a wrapper for Tensorflow, along with common Python libraries such as numpy and pandas. The data was downloaded and saved locally to Google Drive. Mounting the Google Drive and setting runtime to GPU will enable the model to use the uploaded dataset and run effectively.

```python
#Necessary Imports
import os
from sklearn.model_selection import train_test_split
!pip install tf-keras
print("TF_USE_LEGACY_KERAS:", os.getenv("TF_USE_LEGACY_KERAS"))
os.environ['TF_USE_LEGACY_KERAS'] = '1'
print("TF_USE_LEGACY_KERAS:", os.getenv("TF_USE_LEGACY_KERAS"))

try:
  import ktrain
except:
  !pip install ktrain
  import ktrain
from ktrain import text
from ktrain.text import texts_from_df

import pandas as pd
import numpy as np

#Mounting Drive
from google.colab import drive
drive.mount('/content/drive')

#Set runtime to GPU
gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)
```

### 2. Load and Inspect the Data
Using pandas, I loaded the json file and inspected the data header.

```python
#Loading and inspecting the data
reviews = pd.read_json("drive/MyDrive/news_category_trainingdata.json")
reviews.head()
```
<img src="images/dla_head.jpg?raw=true"/> 

### 3. Prepare Data for Modeling
Here, I combined the headline + description into one string for ease of parsing in the machine learning package. Then, I created a "health_wellness" column in the dataset and used numpy to generate values indicating "1" for "Healthy Living" articles and "0" for articles outside of that category. Finally, I used the describe() method on the column to find how many articles in the set are considered "Healthy Living".

```python
# Categorizing healthy living articles for the dataset
reviews['combined_text'] = reviews['headline'] + ' ' + reviews['short_description']
reviews['healthy'] = np.where((reviews['category'] == 'HEALTHY LIVING'), 1, 0)

reviews['healthy'].describe()
```
<img src="images/dla_describe1.jpg?raw=true"/>

From this result, I can conclude that I need to balance the data, as the mean is much closer to 0 than 1, indicating there are very few "healthy" (~3%) articles relative to the number of "not healthy" articles.

### 4. Balancing the Data
I then generated a balanced set by setting half of the set length to all of the "healthy" articles (sample_amount) and populated the set with an equal amount of articles that are not considered "healthy" (not_healthy).

```python
#Total number of healthy articles
sample_amount =  len(reviews[reviews["healthy"] == 1])

health_wellness = reviews[reviews['healthy'] == 1].sample(n=sample_amount)
not_health_wellness = reviews[reviews['healthy'] == 0].sample(n=sample_amount)

#Total reviews
review_sample = pd.concat([health_wellness,not_health_wellness])

#Describe set
review_sample.describe()
```
<img src="images/dla_describe2.jpg?raw=true"/>

Again, this demonstrates that balancing the data was necessary. The new dataset only contains 13k articles, as opposed to the original 200k.

### 5. Model Type and Optimization Aims
Now with a balanced dataset, I can work on testing and tuning a model to find the best one. For purposes of this project, I used the **distilbert-base-uncased** transformer model. I tuned the dataset and model parameters in an attempt to more accurately categorize "healthy" articles. Given that the aim is to advertise to people reading healthy articles and the cost of advertising to non-healthy article readers is fairly low (they may still purchase the product, they are only less likely to), I would rather have some false positives if it means we capture the majority of the correct audience.

Thus, I am aiming for **maximum recall** and **>80% precision** so that we capture the full healthy living audience with minimal false positives.

```python
#Model parameter setting
train, val, preprocess = ktrain.text.texts_from_df(
    review_sample,
    "combined_text",
    label_columns=["healthy"],
    val_df=None,
    max_features=20000,
    maxlen=512,
    val_pct=0.1,
    ngram_range=1,
    preprocess_mode="distilbert",
    verbose=1
)
```
<img src="images/dla_params.jpg?raw=true"/>

```python
#Model learner tuning
model = preprocess.get_classifier()
learner = ktrain.get_learner(model, train_data=train, val_data=val, batch_size=16)

#Run learning epochs
learner.lr_find(max_epochs=6)
```
<img src="images/dla_epochs.jpg?raw=true"/>

```python
#Learner plot display
learner.lr_plot()
```
<img src="images/dla_learner.jpg?raw=true"/>

Based on the learner plot, I should set the log learning rate to the minimum validation loss at approximately 1e-4. Now, I can use the tuned learner to train the best model. With early stopping, the model should stop sooner than 10 epochs, but I set the limit in case the model does not converge.

```python
#Training model via learner
history=learner.autofit(
    1e-4,
    checkpoint_folder='checkpoint',
    epochs=10,
    early_stopping=True
)
```
<img src="images/dla_training.jpg?raw=true"/>

I then saved the predictor for later reloading, and printed the validation metrics so that I could assess the model's performance.

```python
#Saving predictor for later reloading
predictor_trial_1 = ktrain.get_predictor(learner.model, preproc=preprocess)
predictor_trial_1.save("drive/MyDrive/trial1.healthy_living")

#Validation metrics for training
validation = learner.validate(val_data=val, print_report=True)
```
<img src="images/dla_metrics1.jpg?raw=true"/>

The learner validation score shows that recall and precision are 90% and 85%, respectively, for the "healthy" articles for this set - this indicates that the model was able to identify 90% of the true positives from the dataset and 85% of the total identified as "healthy" were actually "healthy" articles. In order to maximize the hypothetical customer's reach, I would want to increase the model's ability to accurately identify "healthy" articles, which may mean adjusting the training sample to more closely reflect the mix in the total articles.

### 6. Adjusting Sample and Rerunning Model

Here, I used the full amount of "healthy" articles but increased the "non-healthy" sample by 2x, yielding a set that is 33% "healthy" and 67% "non-healthy." I then followed the same steps and training to see how the end result compared.

```python
#Total number of healthy articles
sample_amount =  len(reviews[reviews["healthy"] == 1])

#Create new dataset using the same sample_amount for "healthy" and reducing for
#"non-healthy"
healthy_new = reviews[reviews['healthy'] == 1].sample(n=sample_amount)
not_healthy_new = reviews[reviews['healthy'] == 0].sample(n=round(2*sample_amount))

#Total reviews
review_sample_new = pd.concat([healthy_new,not_healthy_new])

#Describe set
review_sample_new.describe()
```
<img src="images/dla_describe3.jpg?raw=true"/>

I repeated the same steps as with the first trial and achieved the following results:

```python
#Save for later reloading
predictor_trial_2 = ktrain.get_predictor(learner.model, preproc=preprocess)
predictor_trial_2.save("drive/MyDrive/trial2.healthy_living")

#Validation of new learner
validation = learner.validate(val_data=val, print_report=True)
```
<img src="images/dla_metrics2.jpg?raw=true"/>

Given these two results, it seemed that the sample composition needed further adjustment. I ran a third trial with 45% non-healthy and 55% healthy articles.

```python
#Save for later reloading
#Total number of healthy articles
sample_amount =  len(reviews[reviews["healthy"] == 1])

#Create new dataset using the same sample_amount for "healthy" and reducing for
#"non-healthy"
healthy_new = reviews[reviews['healthy'] == 1].sample(n=sample_amount)
not_healthy_new = reviews[reviews['healthy'] == 0].sample(n=round(0.8*sample_amount))

#Total reviews
review_sample_new = pd.concat([healthy_new,not_healthy_new])

#Describe set
review_sample_new.describe()
```
<img src="images/dla_describe4.jpg?raw=true"/>

And used the same process as the first trial, getting the following results:

```python
#Save for later reloading
predictor_trial_3 = ktrain.get_predictor(learner.model, preproc=preprocess)
predictor_trial_3.save("drive/MyDrive/trial3.healthy_living")

#Validation of new learner
validation = learner.validate(val_data=val, print_report=True)
```
<img src="images/dla_metrics3.jpg?raw=true"/>

### Conclusion

Based on these three trials, it would seem the 55% set has the most balanced improvement, with both accuracy and test error rate not reaching such a high level that the model is overfitted. The fact that the model reaches an f1-score of 90% for positive findings is also encouraging, as it ensures that the model can accurately and precisely find 90% of the articles that the hypothetical customer would like to advertise to.

