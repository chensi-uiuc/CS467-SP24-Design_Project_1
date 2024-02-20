import os  # For interacting with the operating system
import pandas as pd  # For data manipulation and analysis
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS  # For text feature extraction
import plotly.graph_objs as go  # For creating interactive plots
import numpy as np  # For numerical operations

# Function to read emails from a specified directory and return them as a list
def read_emails_from_directory(directory_path):
    emails = [] 
    for filename in os.listdir(directory_path):  # Iterate over each file in the directory
        filepath = os.path.join(directory_path, filename)  
        if os.path.isfile(filepath):  
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as email_file:  
                emails.append(email_file.read())  # Append the content of the file to the emails list
    return emails  # Return the list of email contents

# Function to process datasets of spam and ham emails
def process_dataset(spam_dir, ham_dir):
    spam_emails = read_emails_from_directory(spam_dir)  
    ham_emails = read_emails_from_directory(ham_dir) 
    emails = spam_emails + ham_emails  
    additional_stop_words = ['your', 'yours', 'ours'] 
    stop_words = list(ENGLISH_STOP_WORDS.union(additional_stop_words)) 
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)  # Initialize the TF-IDF vectorizer with stop words
    tfidf_matrix = tfidf_vectorizer.fit_transform(emails)  # Vectorize the email content and get the TF-IDF matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()  # Get the feature names (words)
    sum_tfidf = np.sum(tfidf_matrix, axis=0)  # Sum the TF-IDF values for each word across all emails
    sum_tfidf = np.squeeze(np.asarray(sum_tfidf))  # Convert the sum to a numpy array
    df_tfidf = pd.DataFrame({'Word': feature_names, 'Summed TF-IDF': sum_tfidf})  # Create a DataFrame with words and their summed TF-IDF
    spam_count, ham_count = len(spam_emails), len(ham_emails)  # Count the number of spam and ham emails
    # Create binary matrices for spam and ham to count term frequencies
    binary_matrix_spam = (tfidf_matrix[:spam_count, :].todense() > 0)
    binary_matrix_ham = (tfidf_matrix[spam_count:, :].todense() > 0)
    # Add spam and ham frequency columns to the DataFrame
    df_tfidf['Spam Frequency'] = np.array(binary_matrix_spam.sum(axis=0)).flatten()
    df_tfidf['Ham Frequency'] = np.array(binary_matrix_ham.sum(axis=0)).flatten()
    # Return the top 50 terms sorted by their summed TF-IDF values
    return df_tfidf.sort_values(by='Summed TF-IDF', ascending=False).head(50)

# Define directories containing datasets of spam and ham emails
datasets = [['data/enron1/ham', 'data/enron1/spam'],
            ['data/enron2/ham', 'data/enron2/spam'],
            ['data/enron3/ham', 'data/enron3/spam'],
            ['data/enron4/ham', 'data/enron4/spam'],
            ['data/enron5/ham', 'data/enron5/spam'],
            ['data/enron6/ham', 'data/enron6/spam']]
# Process each dataset and store the resulting DataFrames
dataframes = [process_dataset(i[0], i[1]) for i in datasets]

# Initialize a Plotly Figure for plotting
fig = go.Figure()

# Add x=y dashed line to the figure for visual reference
for i in range(len(dataframes)):
    fig.add_trace(
        go.Scatter(
            x=[0, max(dataframes[i]['Spam Frequency'].max(), dataframes[i]['Ham Frequency'].max())],
            y=[0, max(dataframes[i]['Spam Frequency'].max(), dataframes[i]['Ham Frequency'].max())],
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
            x=df['Spam Frequency'], y=df['Ham Frequency'], mode='markers',  # Scatter plot with markers
            marker=dict(size=12), name=f'Dataset {i+1}', 
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
        label=f"Dataset {i+1}",  
        method="update", 
        args=[{"visible": [False] * len(dataframes)},  # Initial visibility state
              {"title": f"Term Frequencies in Spam vs. Ham Emails: Dataset {i+1}"}]  # Update plot title on click
    ) for i in range(len(dataframes))
]

# Set each button to make its corresponding dataset visible
for i, _ in enumerate(buttons):
    buttons[i]['args'][0]['visible'][i] = True

# Update the figure layout with the buttons and set initial plot title and hover mode
fig.update_layout(
    updatemenus=[dict(active=0, buttons=buttons, x=0.0, xanchor='left', y=1.2, yanchor='top')],  # Position and add buttons
    title="Term Frequencies in Spam vs. Ham Emails: Dataset 1",  # Initial plot title
    hovermode='closest'  # Hover mode setting
)

# Display the figure
fig.show()
