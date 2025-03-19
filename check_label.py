import pandas as pd

df = pd.read_csv("drawing_data.csv")
print(df["label"].value_counts())  # Show number of Human (0) and Bot (1) entries
