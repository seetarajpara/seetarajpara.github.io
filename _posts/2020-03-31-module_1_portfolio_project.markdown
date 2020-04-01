---
layout: post
title:      "Module 1 Portfolio Project"
date:       2020-04-01 01:08:56 +0000
permalink:  module_1_portfolio_project
---


This project gave me the opportunity to delve deeper into a dataset that details the housing prices within King County, Washington. In this blog post, I will discuss my analysis and rationale behind the analysis for each particular variable that could influence housing prices in King County.

First, I wanted to get an idea of what the variables are. I ran a quick script to get the gist of the variables I'd be dealing with and found this: 

`df_housedata.columns`

```Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'],
      dtype='object')```
			
		
From there, I wanted to understand how these variables relate to the price of the house. I went ahead and made a correlation heatmap (based on Pearson correlation values) to see which variables are closely related to each other:
	
```
# correlation matrix
corrmat=df_housedata.corr()
f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corrmat,vmax=1.0,square=True)
```

https://imgur.com/x56MgQl

With that, I decided to get some scatter plots showing the quantitative relationship between specific variables, like sqft_living vs. price:

```
var='sqft_living'
data=pd.concat([df_housedata['price'], df_housedata[var]],axis=1)
data.plot.scatter(x=var, y='price',ylim=(0,8000000));
```

https://imgur.com/swbArxg

I went further and used boxplots to visualize categorical data vs. price (Note: I used this same code snippet and adjusted the variable to make the following plots):  

```
grade='grade'
data=pd.concat([df_housedata['price'], df_housedata[grade]], axis=1)
f, ax = plt.subplots(figsize=(7,7))
fig = sns.boxplot(x=grade, y='price', data=data)
fig.axis(ymin=0, ymax=8000000);
```

https://imgur.com/BDZ0Ofu
https://imgur.com/T3Da3sH
https://imgur.com/L6aEYgB
https://imgur.com/DVQPB3i
https://imgur.com/41KWxJl
https://imgur.com/ZLR1poK

These plots show a really obvious correlation with the following variables:
* Bedrooms
* Bathrooms
* Grade
* Floors
* Square footage of living area
* Zip code

The year built variable surprisingly did not illustrate a correlation with price. Furthermore, a nice summary of this information can also be inferred using the following code: 

```
sns.set()
cols=['price','sqft_living','bedrooms','bathrooms', 'grade', 'yr_built','floors']
sns.pairplot(df_housedata[cols],size=2.5)
plt.show()
```

https://imgur.com/AeLtdwb

In summary, the analysis of the King County Housing Prices dataset show that these variables influence the price of a home in that particular county:
1. Square footage of living area
2. Number of bedrooms 
3. Number of bathrooms
4. Floors
5. Zip code
6. Grade


