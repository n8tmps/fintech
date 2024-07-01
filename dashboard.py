import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

##########################
# PAGE SETUP
##########################
st.set_page_config(page_title="TSLA Financials", page_icon=":bar_chart:", layout="wide")
st.markdown('<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Tesla_logo.png/900px-Tesla_logo.png" width="100">', unsafe_allow_html=True)
st.title("Tesla Financials Dashboard")

# Function to load data
def load_data(path):
    data = pd.read_csv(path)
    return data

# Function to create the stacked bar chart
def create_stacked_bar_chart(data, metrics, date_order):
    # Filter the DataFrame for the selected metrics
    filtered_df = data[data['Metric'].isin(metrics)]

    # Melt the DataFrame to long format
    filtered_df = pd.melt(filtered_df, id_vars=['Metric'], var_name='Date', value_name='Value')

    # Convert 'Value' column to numeric
    filtered_df['Value'] = pd.to_numeric(filtered_df['Value'], errors='coerce')

    # Create pivot table for stacking
    pivot_df = filtered_df.pivot(index='Date', columns='Metric', values='Value')

    # Reindex the pivot table to match the custom order
    pivot_df = pivot_df.reindex(date_order)

    # Plot the stacked bar chart using Plotly
    fig = go.Figure()

    for metric in metrics:
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[metric],
            name=metric
        ))

    # Update layout for better appearance
    fig.update_layout(
        barmode='stack',
        title='Annual Financial Metrics',
        xaxis_title='Date',
        yaxis_title='Value',
        yaxis=dict(tickmode='array',
                   tickvals=[0, 50e6, 100e6, 150e6, 200e6],
                   ticktext=['0', '$50M', '$100M', '$150M', '$200M'], 
                   showgrid=False),
        legend_title='Metric',
        template='plotly_white'
    )

    # Add text annotation in the bottom right corner
    fig.add_annotation(
        x=1,
        y=-0.2,
        xref='paper',
        yref='paper',
        text='Source: Yahoo Finance TSLA Financials Income Statement',
        showarrow=False,
        xanchor='right',
        yanchor='top',
        font=dict(size=12, color='gray'),
        align='right'
    )

    return fig

# Function to create the line plot for quarterly revenue trends
def create_line_plot(quarters, revenue):
    # Create the line plot using Plotly
    fig_line_plot = go.Figure()

    # Plot the line plot for quarterly revenue
    fig_line_plot.add_trace(go.Scatter(
        x=quarters,
        y=revenue,
        mode='lines+markers',
        name='Total Revenue'
    ))

    # Update layout for line plot
    fig_line_plot.update_layout(
        title='Quarterly Revenue Trends from Q1 2019 to Q1 2024',
        xaxis_title='Quarter',
        yaxis_title='Revenue',
        font=dict(size=12),
        xaxis=dict(tickangle=45),
        yaxis=dict(tickmode='array',
                   tickvals=[0, 5e6, 10e6, 15e6, 20e6, 25e6],
                   ticktext=['0', '$5M', '$10M', '$15M', '$20M', '$25M'], 
                   showgrid=False),
    )

    # Add text annotation in the bottom right corner
    fig_line_plot.add_annotation(
        x=1,
        y=-0.2,
        xref='paper',
        yref='paper',
        text='Source: Tesla\'s Investor Relations website',
        showarrow=False,
        xanchor='right',
        yanchor='top',
        font=dict(size=12, color='gray')
    )

    return fig_line_plot

# Load the data
df = load_data("./incomestatement.csv")
tesla_data = load_data("./Updated_Tesla_Quarterly_Revenue.csv")

# Define the desired metrics
all_metrics = ['EBITDA', 'Total Revenue', 'Cost of Revenue']

# Filter data for the desired range
filtered_df = df[df['Metric'].isin(all_metrics)]

# Define the custom order for the dates
date_order = ['12/31/2020','12/31/2021', '12/31/2022', '12/31/2023', 'TTM']

# Create the stacked bar chart
fig_bar_chart = create_stacked_bar_chart(df, all_metrics, date_order)

# Data for Plotly line plot
quarters = tesla_data['Quarter']
revenue = tesla_data['Total Revenue']

# Filter data for the desired range
filtered_quarters = quarters.iloc[-21:]
filtered_revenue = revenue.iloc[-21:]

# Create the line plot
fig_line_plot = create_line_plot(filtered_quarters, filtered_revenue)


### STREAMLIT LAYOUT ####
top_left_column, top_right_column = st.columns((2, 1))
bottom_left_column, bottom_right_column = st.columns(2)

with bottom_left_column:
    st.plotly_chart(fig_bar_chart)

with bottom_right_column:
    st.plotly_chart(fig_line_plot)


with top_left_column:
    # Create columns for each metric
    col1, col2, col3 = st.columns(3)
    col1.metric("EBITDA", "$13,796,000", "-0.07%")
    col2.metric("Total Revenue", "$94,745,000", "-0.02%")
    col3.metric("Cost of Revenue", "$77,900,000", "-0.01%")
    

# Rename the columns
df = df.rename(columns={
    'TTM': '2024',
    '12/31/2023': '2023',
    '12/31/2022': '2022',
    '12/31/2021': '2021',
    '12/31/2020': '2020'
})

# Convert numeric values to appropriate types (int or float) while retaining the "Metric" column
numeric_cols = df.columns[1:]  # Exclude the "Metric" column
df[numeric_cols] = df[numeric_cols].map(lambda x: pd.to_numeric(x.replace(',', ''), errors='coerce') if isinstance(x, str) else x)

# Apply formatting with commas for both integers and floats, and parentheses for negative values
df[numeric_cols] = df[numeric_cols].map(lambda x: f"({abs(x):,.2f})" if isinstance(x, float) and x < 0 else (f"({abs(int(x)):,})" if isinstance(x, int) and x < 0 else (f"{x:,.2f}" if isinstance(x, float) else (f"{x:,}" if isinstance(x, int) else x))))

# Replace 'nan' with '---'
df = df.replace('nan', '---')

# Convert numeric values back to strings
df[numeric_cols] = df[numeric_cols].astype(str)

# Display the dataframe in Streamlit with full width using CSS
st.write("""
    <style>
        .dataframe {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

st.write("Detailed Income Statement")
st.dataframe(df, hide_index=True)







