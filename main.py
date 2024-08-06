import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

########
# DATA #
########

os.makedirs('data', exist_ok=True)

# Load the data
gdp = pd.read_csv('data/gdp.csv', index_col='DATE', parse_dates=True)
unemployment_rate = pd.read_csv('data/unemployment_rate.csv', index_col='DATE', parse_dates=True)
inflation_rate = pd.read_csv('data/inflation_rate.csv', index_col='DATE', parse_dates=True)
inflation_rate.rename(columns={'FPCPITOTLZGUSA': 'Inflation Rate'}, inplace=True)
sp500 = yf.download('^GSPC', start='1960-01-01', end='2024-01-01')['Adj Close']

# Resample the data
gdp_monthly = gdp.resample('M').ffill()
unemployment_rate_monthly = unemployment_rate.resample('M').mean()
inflation_rate_monthly = inflation_rate.resample('M').ffill()
sp500_monthly = sp500.resample('M').ffill()
sp500_returns = sp500.pct_change().dropna() * 100  # Convert to percentage

# Align the datasets
start_date = max(gdp_monthly.index.min(), unemployment_rate_monthly.index.min(), inflation_rate_monthly.index.min(), sp500_monthly.index.min())
end_date = min(gdp_monthly.index.max(), unemployment_rate_monthly.index.max(), inflation_rate_monthly.index.max(), sp500_monthly.index.max())

gdp_filtered = gdp_monthly.loc[start_date:end_date]
unemployment_rate_filtered = unemployment_rate_monthly.loc[start_date:end_date]
inflation_rate_filtered = inflation_rate_monthly.loc[start_date:end_date]
sp500_filtered = sp500_monthly.loc[start_date:end_date]
sp500_returns_filtered = sp500_returns.loc[start_date:end_date]

#############
# STREAMLIT #
#############

st.title("Economic Indicator Dashboard")

# Sidebar for date range selection
start_date = st.sidebar.date_input('Start date', min_value=gdp_filtered.index.min(), max_value=gdp_filtered.index.max(), value=gdp_filtered.index.min())
end_date = st.sidebar.date_input('End date', min_value=gdp_filtered.index.min(), max_value=gdp_filtered.index.max(), value=gdp_filtered.index.max())

# Filter data
gdp_filtered = gdp_monthly.loc[start_date:end_date]
unemployment_rate_filtered = unemployment_rate_monthly.loc[start_date:end_date]
inflation_rate_filtered = inflation_rate_monthly.loc[start_date:end_date]
sp500_filtered = sp500_monthly.loc[start_date:end_date]
sp500_returns_filtered = sp500_returns.loc[start_date:end_date]

# Overview Section
st.header("Overview")
st.write("### Summary Statistics")
st.write(f"**GDP**: Latest: {gdp_filtered.iloc[-1, 0]:,.2f}, Average: {gdp_filtered.mean().values[0]:,.2f}, Min: {gdp_filtered.min().values[0]:,.2f}, Max: {gdp_filtered.max().values[0]:,.2f}")
st.write(f"**Unemployment Rate**: Latest: {unemployment_rate_filtered.iloc[-1, 0]:.2f}%, Average: {unemployment_rate_filtered.mean().values[0]:.2f}%, Min: {unemployment_rate_filtered.min().values[0]:.2f}%, Max: {unemployment_rate_filtered.max().values[0]:.2f}%")
st.write(f"**Inflation Rate**: Latest: {inflation_rate_filtered.iloc[-1, 0]:.2f}%, Average: {inflation_rate_filtered.mean().values[0]:.2f}%, Min: {inflation_rate_filtered.min().values[0]:.2f}%, Max: {inflation_rate_filtered.max().values[0]:.2f}%")
st.write(f"**S&P 500 Returns**: Latest: {sp500_returns_filtered.iloc[-1]:.2f}%, Average: {sp500_returns_filtered.mean():.2f}%, Min: {sp500_returns_filtered.min():.2f}%, Max: {sp500_returns_filtered.max():.2f}%")

# GDP Section
st.header("Gross Domestic Product (GDP)")
st.line_chart(gdp_filtered, use_container_width=True)
st.write("### GDP Data")

# Unemployment Rate Section
st.header("Unemployment Rate")
st.line_chart(unemployment_rate_filtered, use_container_width=True)
st.write("### Unemployment Rate Data")

# Inflation Rate Section
st.header("Inflation Rate")
st.line_chart(inflation_rate_filtered, use_container_width=True)
st.write("### Inflation Rate Data")

# S&P 500 Index Section
st.header("S&P 500 Index")
st.line_chart(sp500_filtered, use_container_width=True)
st.write("### S&P 500 Data")

# S&P 500 Returns Section
st.header("S&P 500 Returns")
st.line_chart(sp500_returns_filtered, use_container_width=True)
st.write("### S&P 500 Returns Data")

# Correlation analysis section
combined_data = pd.concat([gdp_filtered, unemployment_rate_filtered, inflation_rate_filtered, sp500_returns_filtered], axis=1)
combined_data.columns = ['GDP', 'Unemployment Rate', 'Inflation Rate', 'S&P 500 Returns']

correlation_matrix = combined_data.corr()

st.header("Correlation Analysis")
st.write("### Correlation Matrix")
st.dataframe(correlation_matrix)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
st.pyplot(fig)
