import pandas as pd
import json

with open('./config.json') as f:
    config = json.load(f)

beginning_year = config["years"][0]
ending_year = config["years"][-1]

dfs = [pd.read_csv(f'data/{year}_movie_collection_data.csv')
       for year in range(beginning_year, ending_year+1)]

# Combine the dataframes
combined_df = pd.concat(dfs)

combined_df.to_csv('./data/full_movie_collection_data.csv', index=False)
