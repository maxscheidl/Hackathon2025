# Hackathon 2025 Bruteforcers

## How to run

To set up this project you first have to download the dataset from orderfox and place all the JSON files in the `hackathon_data/` folder. 

### Quick Start

1. Install Dependencies
2. Load `index.pkl` file into the `index` folder at the top level.
3. Run `streamlit run app.py` to start the interactive UI.

## From Scratch

### Install dependencies

### 1. Data Cleaning

First we performed data cleaning to get rid of unnecessary data. We used the `dataset_filter.py` script to do this.
The script will create go through the files in the `hackathon_data/` folder and put the cleaned data into the
`filtered_data/` folder.

```bash
python dataset_filter.py
```

### 2. Vector store creation

To create the vector store we used the `create_vector_database.py` script. This script will chunk the data and stores it in
a vector store. To make the script run you have to put your OPENAI API key in the corresponding variable in the script.

```bash
python create_vector_database.py
```

### 3. Keyword store creation

The keyword store can be created with the `keyword_dbsqlite.py` script. This script will create a keyword store from the filtered data.

```bash
python keyword_dbsqlite.py
```

### 4. Querying

#### Interactive UI

We provide an interactive UI to query the data. To run the UI you have to run the `app.py` script using streamlit.

```bash
python streamlit run app.py
```


#### Notebook

To run queries we prepared a notebook: `bruterforcers_rag.ipynb`. This notebook will load the vector store and the keyword store and will allow you to run queries on the data.
Hyperparameters can be set at the beginning.
