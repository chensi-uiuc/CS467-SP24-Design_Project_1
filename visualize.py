import os  # For interacting with the operating system
import pandas as pd  # For data manipulation and analysis
import plotly.graph_objs as go  # For creating interactive plots
import numpy as np  # For numerical operations


# Load the list of DataFrames from the HDF5 file
dataframes = []
with pd.HDFStore('Processed_Data.h5', 'r') as store:
    for key in store.keys():
        dataframes.append(store[key])

# Read the list of dataset names
dataset_names = []
with open('dataset_names.txt', 'r') as file:
    dataset_names = [line.strip() for line in file]

# Initialize a Plotly Figure for plotting
fig = go.Figure()

epsilon = 1
# Add x=y dashed line to the figure for visual reference
for i in range(len(dataframes)):
    fig.add_trace(
        go.Scatter(
            # x=[0, max(dataframes[i]['Spam Frequency'].max(), dataframes[i]['Ham Frequency'].max())],
            # y=[0, max(dataframes[i]['Spam Frequency'].max(), dataframes[i]['Ham Frequency'].max())],
            x=[0, max(np.log10(dataframes[i]['Spam Frequency']+epsilon).max(), np.log10(dataframes[i]['Ham Frequency']+epsilon).max())],
            y=[0, max(np.log10(dataframes[i]['Spam Frequency']+epsilon).max(), np.log10(dataframes[i]['Ham Frequency']+epsilon).max())],
            mode='lines',
            line=dict(dash='dash'),  
            name='y=x', 
            showlegend=False, 
            visible=(i == 0)
        )
    )


# Add scatter plots for each dataset to visualize term frequencies
for i, df in enumerate(dataframes):
    fig.add_trace(
        go.Scatter(
            # x=df['Spam Frequency'], y=df['Ham Frequency'],
            x=np.log10(df['Spam Frequency']+epsilon),  # Log-transform Spam Frequency
            y=np.log10(df['Ham Frequency']+epsilon), 
            mode='markers',  # Scatter plot with markers
            # marker=dict(size=12), 
            marker=dict(
                size=12,
                color=np.log(df['Summed TF-IDF']),  # Set color to the "tf-idf" values
                colorscale='viridis',  # Choose the viridis colormap
                colorbar=dict(title='Log TF-IDF', len=0.7),  # Add colorbar with a title
                opacity = 0.7
            ),
            name=dataset_names[i], 
            text=df['Word'],  # Hover text (word)
            customdata=df[['Spam Frequency', 'Ham Frequency', 'Summed TF-IDF']], 
            hovertemplate="<b>%{text}</b><br><br>Spam Frequency: %{customdata[0]}<br>" +
                          "Ham Frequency: %{customdata[1]}<br>Summed TF-IDF: %{customdata[2]}<br>",  # Custom hover text
            visible=(i == 0)  # Only make the first dataset visible by default
        )
    )

# Configure buttons for interactive dataset selection
buttons = [
    dict(
        # label=f"Dataset {i+1}",  
        label=dataset_names[i],
        method="update", 
        args=[{"visible": [False] * len(dataframes)},  # Initial visibility state
              {"title": f"Word Frequencies in Spam vs. Ham Emails: {dataset_names[i]}"}]  # Update plot title on click
    ) for i in range(len(dataframes))
]

# Set each button to make its corresponding dataset visible
for i, _ in enumerate(buttons):
    buttons[i]['args'][0]['visible'][i] = True

# Update the figure layout with the buttons and set initial plot title and hover mode
fig.update_layout(
    updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor='left', y=1.2, yanchor='top')],  # Position and add buttons
    title=f"Word Frequencies in Spam vs. Ham Emails: {dataset_names[0]}",  # Initial plot title
    hovermode='closest',  # Hover mode setting
    xaxis=dict(title='Log10 Spam Frequency'),  # X-axis label
    yaxis=dict(title='Log10 Ham Frequency')  # Y-axis label
)

# Save the figure as an HTML file
fig.write_html("visualization.html")
print("Visualization saved!")

# Display the figure
fig.show()
