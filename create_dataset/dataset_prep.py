from config import *
import pandas as pd

df = pd.read_csv(DATASET_CSV)
df = df.dropna(subset=['audio_file', 'text'])

df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r'\b(avito|Avito)\b', 'авито', case=False)

df.to_csv('updated_dataset.csv', index=False)

print(f"Обработанный датасет сохранен в {DATASET_CSV}")
