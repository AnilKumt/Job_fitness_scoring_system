import pandas as pd

df = pd.read_csv("data/AI_Resume_Screening.csv")

names = df.Name.tolist()
with open('data/names.txt', 'w') as f:
    for item in names:
        f.write(f"{item}\n")

skills = df.Skills.tolist()
