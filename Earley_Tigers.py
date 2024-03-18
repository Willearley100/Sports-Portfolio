#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## We begin by importing the data into a Data Frame. We explore the data using the describe command for statistical information and the isnull command to see if there are any null values. 

# In[3]:


pitcher = pd.read_csv('/Users/williamearley/Detroit Tigers/PitcherData.csv')


# In[29]:


# Making sure we can see all of the columns
pd.set_option('display.max_columns', None)


# In[31]:


pitcher.head(5)


# In[6]:


pitcher.describe()


# In[8]:


pitcher.isnull().sum()


# ## A lot of times it can be helpful to get a visual for the question at hand. Here we see box plots for each different pitch type thrown by the pitcher. We see that the fastball (FB) seems to have the greatest variance, potentially due to the outliers at the top. Let's do some more digging and see why this might be.

# In[39]:


# Box plot to visualize variance in fastball velocities
plt.figure(figsize=(12, 10))
sns.boxplot(x='PitchType', y='ReleaseSpeed', data=pitcher)
plt.title('Variance in Pitch Type Velocities')
plt.xlabel('Pitch Type')
plt.ylabel('Release Speed')
plt.show()


# In[14]:


# Calculate variance in fastball velocities
fastball_variances = pitcher[pitcher['PitchType'] == 'FB']['ReleaseSpeed'].var()

# Generate short report
report = f"The pitcher exhibits high variance in fastball velocities with a variance value of {fastball_variances:.2f}."
print(report)


# In[41]:


pitcher_sorted = pitcher.sort_values(by='ReleaseSpeed', ascending=False)


# ### Below we see the pitches with the 10 fastest release speeds. The top 6 have release speeds above 124 mph and zone speeds above 115 mph. These both seem a bit unrealistic and might imply an error in the radar gun. Six of the ten were fastballs.

# In[43]:


pitcher_sorted.head(10)


# ## Another method we can pursue is feature importance. Here we use a Random Forrest Regressor to identify the features that most impact Release Speed. 

# In[26]:


# For this analysis we only want to know about the fastballs, so we remove the pitches that weren't "FB".
fastball_only = pitcher[pitcher['PitchType'] == 'FB']


# In[33]:


fastball_only.head(5)


# In[34]:


# Select relevant columns from the glossary
relevant_columns = ['ReleaseSpeed', 'ZoneSpeed', 'PitchVerticalApproachAngle', 'ReleaseExtension', 'ReleaseHeight', 'ReleaseSide', 'ArmAngleBR', 'ArmSlotBR', 'StrideLength', 'StrideLengthPercentHeight', 'StrideWidth', 'TimeToPlate', 'PeakHipsVeloX', 'PeakPitchHandVeloX', 'DLHipRotMin', 'HSSepMin', 'PARotMax', 'PAScapRetMin', 'SLKneeFlexFP', 'PAElbowFlexBR', 'SLKneeFlexBR', 'TorsoFBBR', 'TorsoSBBR', 'PeakMomDL', 'PeakMomSL', 'PeakMomTorso', 'PeakMomLFA', 'PeakMomLUA', 'PeakMomPFA', 'PeakMomPUA', 'PeakMomTotalBody', 'PeakPelvisRotVelo', 'PeakPelvisRotVeloTime', 'PeakChestRotVelo', 'PeakChestRotVeloTime', 'PeakElbowExtVelo', 'PeakElbowExtVeloTime', 'PeakShoulderIRVelo', 'PeakShoulderIRVeloTime', 'PelvisChestPeakTimeDiff', 'PeakSLKneeExtVelo', 'ClosingSpeed', 'ClosingTime', 'TimeFirstMove', 'TimeHandSeparation', 'TimePeakKneeLift', 'TimeFootPlant', 'TimeMER']

# Extract relevant data
pitching_data_relevant = fastball_only[relevant_columns].copy()


# In[35]:


from sklearn.ensemble import RandomForestRegressor

# Separate features and target variable
X = pitching_data_relevant.drop('ReleaseSpeed', axis=1)  # Features
y = pitching_data_relevant['ReleaseSpeed']  # Target variable


# ### Because we have some NaN values, we choose to drop these rows and not include them in our model. Another approach here would be the replace the NaN values with the relevant mean values. 

# In[36]:


# Drop rows with missing values
X_no_missing = X.dropna()

# Correspondingly, filter the target variable
y_no_missing = y[X.index.isin(X_no_missing.index)]


# In[37]:


model = RandomForestRegressor()
# Fit the model with the data without missing values
model.fit(X_no_missing, y_no_missing)


# In[38]:


# Get feature importances
feature_importances = model.feature_importances_

# Create a DataFrame to visualize feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances for Fastball Variance')
plt.show()

