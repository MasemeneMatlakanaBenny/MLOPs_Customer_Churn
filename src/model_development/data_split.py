import pandas as pd
from sklearn.model_selection import train_test_split


## load the data
df=pd.read_csv("data/customer_churn.csv")

train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)


train_df.to_csv("train_data.csv")
test_df.to_csv("test_data.csv")
