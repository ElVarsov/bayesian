import pandas as pd

df = pd.read_csv('mortality_data.csv' , index_col=False, skipinitialspace=True)
df['Year'] = df['Year'].astype(int)
print(df['Year'].dtype)
cleaned_df = df[df['Year'] >= 2000]
cleaned_df = cleaned_df[cleaned_df['Age Group'] != '[Unknown]']
print(cleaned_df.head())
cleaned_df.to_csv('cleaned_mortality_data.csv', index=False)