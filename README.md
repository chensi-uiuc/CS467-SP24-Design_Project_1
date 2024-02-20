# CS467-SP24-Design_Project_1

## Project ScamSpeech

Motivation: Our objective is to create visual representations of frequently used words to enable rapid identification of spam emails at first glance. Such visualizations enable both machines and humans to quickly discern patterns and characteristics typical of spam, facilitating rapid classification even before delving into deeper content analysis. This visualization acts as a powerful tool for monitoring and managing email threats, offering a clear and immediate understanding of spam trends and tactics.   
  
### How to run the project: 
#### 1. Open terminal in the folder;
#### 2. Run the following command to analyze the data:
```{python}
>>> python data_procees.py (args)
```
Use the following args to adjust:

| Args   	| Type 	| Description                                                         	|
|--------	|------	|---------------------------------------------------------------------	|
| --n    	| int  	| The top N words with highest Summed TF-IDF scores would get stored. 	|
| --d    	| str  	| The address for the folder of data.                                 	|
| --norm 	| bool 	| Set True to normalize all TF-IDF scores by email.                   	|


For example, a recommended setting is:
```
>>> python data_procees.py --n 500 --d data --True 
```

You should see the preprocessed dataframes getting stored in `Processed_Data.h5`.

#### 3. Run the following command to visualize the data:
```
>>> python visualize.py
```

A new webpage should pop up and the html form of the visualization should be stored as `visualization.html`.
