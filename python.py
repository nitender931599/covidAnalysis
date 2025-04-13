import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
original=pd.read_csv('Provisional_COVID-19_Deaths_by_Sex_and_Age.csv')

# creating a copy of original dataset
df=original.copy()
print("Shape of the dataSet" ,df.shape)
print(df.info())
print(df.describe())

print(df.isnull().sum())
month_missing_per=(df['Month'].isnull().sum()/df.shape[0])*100
#month_missing_per is 10% and dataset is large so we can drop these rows which will not impact our analysis
df=df[~df['Month'].isnull()]
df.shape

c=['COVID-19 Deaths','Total Deaths','Pneumonia Deaths','Pneumonia and COVID-19 Deaths','Influenza Deaths','Pneumonia, Influenza, or COVID-19 Deaths']
r=df[~df['Footnote'].isnull()].index
df.loc[r, c]=df.loc[r,c].fillna(5)
print(df.isnull().sum())
# missing values in footnote is not handled to keep the originality of the data


# Group by Age and Cause of Death
age_grouped = df.groupby(['Group'])[['COVID-19 Deaths', 'Pneumonia Deaths', 'Influenza Deaths']].sum().reset_index()

# Plot comparison
age_grouped.plot(x='Group', kind='bar', stacked=True, figsize=(10, 6))
plt.title('COVID-19 vs Pneumonia vs Influenza Deaths by Age Group')
plt.ylabel('Number of Deaths')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


corr_df = df[['COVID-19 Deaths', 'Pneumonia Deaths', 'Influenza Deaths']].corr()
sns.heatmap(corr_df, annot=True, cmap="coolwarm")
plt.title("Correlation Between COVID-19 and Respiratory Deaths")
plt.show()


plt.figure(figsize=(10,5))
sns.pairplot(df[['COVID-19 Deaths', 'Pneumonia Deaths', 'Influenza Deaths']])
plt.show()

from scipy import stats

# Z-score for outlier detection
df['z_score'] = stats.zscore(df['COVID-19 Deaths'].fillna(0))

# Outliers: z-score > 3
outliers = df[df['z_score'] > 3]
print("Unusually high COVID-19 death counts:")
print(outliers[['State', 'Group', 'COVID-19 Deaths']])


# Assume we have a population file or use static dict for simplicity
import plotly.express as px
state_pop = {'California': 39500000, 'Texas': 29000000, 'Florida': 21500000}  # etc.
df['Population'] = df['State'].map(state_pop)
df['COVID-19 Deaths Per 100k'] = (df['COVID-19 Deaths'] / df['Population']) * 100000

state_level = df.groupby('State')['COVID-19 Deaths Per 100k'].sum().reset_index()

# Geospatial map
fig = px.choropleth(state_level, locations='State', locationmode="USA-states",
                    color='COVID-19 Deaths Per 100k', scope="usa",
                    color_continuous_scale="Reds", title='COVID-19 Deaths Per 100k by State')
fig.show()






# Example dataset (Replace this with your actual dataset)
# Load your actual data like: df = pd.read_csv('your_data.csv')
data = {
    'State': ['California'] * 10 + ['Texas'] * 10,
    'Start week': pd.date_range(start='2020-03-01', periods=10, freq='W')\
                  .tolist() + pd.date_range(start='2020-03-01', periods=10, freq='W').tolist(),
    'COVID-19 Deaths': [5, 10, 50, 100, 150, 180, 100, 60, 30, 10,
                        2, 5, 20, 60, 120, 140, 90, 40, 15, 5]
}
df = pd.DataFrame(data)

# Group by State and Date
state_time = df.groupby(['State', 'Start week'])['COVID-19 Deaths'].sum().reset_index()

# Identify peak periods per state (e.g., top 10% highest death weeks per state)
thresholds = state_time.groupby('State')['COVID-19 Deaths'].quantile(0.90)
state_time['Is Peak'] = state_time.apply(lambda row: row['COVID-19 Deaths'] > thresholds[row['State']], axis=1)

# Plotting death trends with peak markers
plt.figure(figsize=(14, 6))
for state in state_time['State'].unique():
    subset = state_time[state_time['State'] == state]
    plt.plot(subset['Start week'], subset['COVID-19 Deaths'], label=f'{state} - Deaths')
    plt.scatter(subset[subset['Is Peak']]['Start week'],
                subset[subset['Is Peak']]['COVID-19 Deaths'],
                color='red', label=f'{state} - Peak' if state == state_time['State'].unique()[0] else "")

plt.title("COVID-19 Death Trends with High Mortality Peaks by State")
plt.xlabel("Week")
plt.ylabel("Deaths")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

#obj-> 5
# National trend
nation_time = df.groupby('Start week')['COVID-19 Deaths'].sum().reset_index()

# Rolling average
nation_time['7-day Avg'] = nation_time['COVID-19 Deaths'].rolling(4).mean()

# Plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=nation_time, x='Start week', y='COVID-19 Deaths', label='Weekly Deaths')
sns.lineplot(data=nation_time, x='Start week', y='7-day Avg', label='4-week Moving Avg')
plt.title("COVID-19 Death Trend Over Time")
plt.xticks(rotation=45)
plt.show()
