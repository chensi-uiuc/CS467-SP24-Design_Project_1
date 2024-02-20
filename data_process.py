import os  # For interacting with the operating system
import pandas as pd  # For data manipulation and analysis
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS  # For text feature extraction
from sklearn.preprocessing import normalize
import plotly.graph_objs as go  # For creating interactive plots
import numpy as np  # For numerical operations
import argparse

# Function to read emails from a specified directory and return them as a list
def read_emails_from_directory(directory_path):
    emails = [] 
    for filename in os.listdir(directory_path):  # Iterate over each file in the directory
        filepath = os.path.join(directory_path, filename)  
        if os.path.isfile(filepath):  
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as email_file:  
                emails.append(email_file.read())  # Append the content of the file to the emails list
    return emails  # Return the list of email contents


'''
Function to process datasets of spam and ham emails
Input: 
    spam_dir: Dir of spam emails
    ham_dir: Dir of ham emails
    N: Number of words output with highest TF-IDF scores
    normalize: whether to normalize
Output:
    Dataframe of top N words output with highest TF-IDF scores in the dataset
''' 
def process_dataset(spam_dir, ham_dir, N, normalize):
    # Read emails from dataset
    spam_emails = read_emails_from_directory(spam_dir)  
    ham_emails = read_emails_from_directory(ham_dir) 
    emails = spam_emails + ham_emails  

    # manually define stopwords
    additional_stop_words = ['your', 'yours', 'ours', 'subject'] 
    stop_words = list(ENGLISH_STOP_WORDS.union(additional_stop_words)) 

    # Run TF_IDF analysis
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words)  # Initialize the TF-IDF vectorizer with stop words
    tfidf_matrix = tfidf_vectorizer.fit_transform(emails)  # Vectorize the email content and get the TF-IDF matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()  # Get the feature names (words)

    if normalize:
        normalized_td_idf = normalize(tfidf_matrix, norm='l1', axis=1)  # Normalize words in each email 
    else:
        normalized_td_idf = tfidf_matrix

    sum_tfidf = np.sum(normalized_td_idf, axis=0)  # Sum the TF-IDF values for each word across all emails
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
    df_tfidf = df_tfidf[~df_tfidf['Word'].str.isnumeric()]
    return df_tfidf.sort_values(by='Summed TF-IDF', ascending=False).head(N)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your script")
    
    # Define command-line arguments
    parser.add_argument('--n', type=int, help='Num of words with highest TF-IDF')
    parser.add_argument('--d', type=str, help='Address of datasets')
    parser.add_argument('--norm', type=bool, help='Ture to normalize')

    # Parse the command-line arguments
    args = parser.parse_args()

    N = args.n

    # Define directories containing datasets of spam and ham emails
    directory = args.d
    items = os.listdir(directory)
    # Filter out only the folders
    folders = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    datasets = [[directory+'/'+item+'/spam', directory+'/'+item+'/ham'] for item in folders]

    # Writing to a text file
    with open('dataset_names.txt', 'w') as file:
        for string in folders:
            file.write(f"{string}\n")

    # Process each dataset and store the resulting DataFrames
    dataframes = [process_dataset(i[0], i[1], N, normalize) for i in datasets]

    # Save the list of DataFrames to an HDF5 file
    with pd.HDFStore('Processed_Data.h5', 'w') as store:
        for i, df in enumerate(dataframes):
            store[f'df_{i}'] = df

    print("Finished Preprocessing")
