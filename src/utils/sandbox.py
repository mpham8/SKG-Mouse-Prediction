import pandas as pd

# Create example DataFrame with index column
df = pd.DataFrame({
    'Age': [25, 30, 25, 40],
    'poop': [25, 30, 30, 40]
}, index=[1, 2, 1, 4])

# Group by index column and sum the values of the other columns
df = df.groupby(df.index).sum()

# # Reset index to regular column
# df = df.reset_index()

print(df)