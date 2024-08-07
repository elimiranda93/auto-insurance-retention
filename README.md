<div align="center">
  
# Auto Insurance Retention Capstone

</div>

## 🎯 Project Overview

The goal of this project is to build a machine learning model that will identify the likelihood of customer retention based on aspects of their demographic. Identifying aspects of customer demographics associated with increased retention will help to guide insurance carriers' marketing efforts by providing them with insights into what markets they can cut ad spending. This cut in ad spending could be forwarded to customers in other markets in the form of a premium reduction. 

## The Problem Area

My area of interest is property & casualty insurance (insurance associated with property and liability). This project could have broad implications as there are multiple lines and types of insurance with varying challenges. However, for this project I will narrow the scope to personal lines' auto insurance, which contains policies that are purchased and held by a common person for their personal/private automobile. The challenges and/or opportunities in my project would be to identify what demographics in a customer influence them to retain their auto insurance policy. My project’s insights can be used by insurance carriers to identify markets of focus based on demographics associated with high-risk of churn. For example, they can calibrate underwritten premiums for certain customers that have high churn demographics in order to retain them. If an insurance carrier successfully retains customers, they can choose to spend less on ads as they do not need to be concerned with replacing customers due to churn. These ad savings can also be passed on to customers in the form of a premium decrease. 

## 📊 Dataset

The dataset comes from **Kaggle**, a prominent platform for public and paid datasets. See the data dictionary below for information regarding columns in the dataset and the row values they carry. Raw data can be found in the following link: https://drive.google.com/drive/folders/1huMm9Sb7muOWKaOKiQN6lG2Kt61L9U58?usp=sharing
![image](https://github.com/elimiranda93/auto-insurance-retention/assets/166231821/5bbb02b9-ed49-494b-895c-c0480bee572a)


## 🚀 Project Workflow

### ➡️ Data Cleaning
Data cleaning consists of removing irrelevant features, handling null/missing data, converting features into appropriate dtypes and checking for duplicate data. Clean data can be found in the following link(see the file with 'clean' in its name):https://drive.google.com/drive/folders/1tK2drVlQC_fwGeblF-jODaq3gtiQxPOo?usp=sharing

### 🕵️‍♂️ Exploratory Data Analysis (EDA)
EDA will help the understanding of feature distributions and relationship retention and/or churn. This understanding will in turn identify the ideal model choices and guide feature engineering.

## Impact
I anticipate my project will add value in both the business and societal sense. Where insurance carriers can increase customer retention by 5% allowing for a reduction in added spending by 10% or $165,000,000 (Vigderman 2023). These savings in ad spending can be passed on to customers in the form of a reduction in the annual premium by 4%. If we are considering the national average cost of $2,150 annually, we would see the cost be reduced to $2,064 (Martin, Smith 2024). It may not seem like a large amount of savings, but if the machine learning from this project can be adopted by all insurance carriers that carry personal auto insurance, then these savings can be felt by the ~232.8 million licensed drivers in the United States. 

Sources: 
Car Insurance Company Ad Spending | AutoInsurance.com. (2023, March 23). Autoinsurance.com. https://www.autoinsurance.com/research/car-insurance-ads/
Martin, E. (2024, June 11). Car Insurance Facts And Statistics 2024. Forbes Advisor. https://www.forbes.com/advisor/car-insurance/car-insurance-facts-and-statistics/

### 🧠 Feature Engineering
During feature engineering, dummy values were generated for the 'county' and 'marital_status' columns, which allowed us to convert it from a categorical column to a numerical column that can be used for modeling. 

Then the following columns were reviewed for potential feature engineering:

length_of_residence: While exploring this column, the value 6.801000118255615 was observed. This was an odd value considering the rest of the values in this column were whole numbers. All the rows with this value were dropped. 

income: This column was reviewed. However, no changes were made to it as all the row values appeared reasonable. 

home_market_value: This column was categorical and could not be processed in a model in this state. Since the values in this column were a range, the ranges were split into upper and lower bound columns. Then each row-level upper and lower bound values were averaged to create a new numerical avg_home_market_value column. 

age_in_years: This column had a broad range of ages. Running models on such a broad range could be computationally expensive and taxing, so the ages were consolidated into ranges. A concern regarding the quality of data for this column was raised in that there was a significantly higher count of people age 55 years than any other age in the distribution. However, they were kept in the dataframe for this iteration.

### 🏭 Modeling
The testing and evaluation of models includes:

- Logistic Regression: Due to concerns with the data quality and weak correlations observed in heatmaps, the expectations for models on this dataframe are low.
  
- Log 1 Results:
Train Score: 0.3087307059884368
Test Score: 0.30086633663366336
The results were as low as expected. In an effort to improve the performance of the model, the county columns were removed as they showed to have little to no relationship with the target variable during heatmap analysis. 

- Log 2 Results:
Train Score: 0.30395692993157586
Test Score: 0.30129950495049507
The results of this model were low too, with marginal differences from the Log 1 results. 

- XG Boost:
Mean Squared Error: 5111814.5000
R^2 Score: 0.0646
Based on the Mean Squared Error, the XG Boost model is not very good at making predictions on this DataFrame. 
The R^2 score supports our findings above as the model is only able to explain about 7.05% of the variance in the target variable. In other words, the model does not fit this data. 

- Decision Tree:
Cross-validation scores: [0.1170777  0.12027583 0.11788887 0.11709322 0.11616496]
Mean cross-validation score: 0.1177001172868791
The Mean cross-validation score is considerably low. This is likely due to the data quality. However, hyperparameter tuning could be helpful. 

## 📝 Findings and Conclusions
The models in their current state are doing a poor job of making predictions on this data set. The primary cause for this is likely poor data quality. 

Potential improvements:
Conducting research for data with better quality could be a possible solution to the performance of the models for this project. Instead of using generated data, real world data would be ideal or perhaps generated data that was more reflective of the real world. 



