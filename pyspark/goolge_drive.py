# Step 1: Acquire Titanic data from Seaborn and store it in a DataFrame
import seaborn as sns
import pandas as pd

titanic = sns.load_dataset("titanic")

# Step 2: Store the data in titanic.csv format in Google Drive
titanic.to_csv('/content/drive/My Drive/titanic.csv', index=False)
