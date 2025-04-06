# Hackathon 2025 Bruteforcers

## How to run

To set up this project you first have to download the dataset from orderfox and place all the JSON files in the `hackathon_data/` folder. 

### Install dependencies

### Data Cleaning

First we performed data cleaning to get rid of unnecessary data. We used the `dataset_filter.py` script to do this.
The script will create go through the files in the `hackathon_data/` folder and put the cleaned data into the
`filtered_data/` folder.

Simply run:
```bash
python dataset_filter.py
```

### Vector store creation

To create the vector store we used the `create_vector_database.py` script. This script will chunk the data and stores it in
a vector store. To make the script run you have to put your OPENAI API key in the corresponding variable in the script.

Simply run:
```bash
python create_vector_database.py
```

### Keyword store creation

### Querying

To run queries we prepared a notebook: `bruterforcers_rag.ipynb`. This notebook will load the vector store and the keyword store and will allow you to run queries on the data.
Hyperparameters can be set at the beginning.