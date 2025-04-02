#%%
import pandas as pd
import google.generativeai as genai
import time
#%%
genai.configure(api_key="GENAI_API_KEY")
#%%
data = pd.read_csv('src/dataset/Financial-QA-10k.csv')
#%%
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)
#%%
def generate_messages(row: pd.Series) -> dict:
    return {
        'text_input': str(row['question']),
        'output': str(row['answer'])
    }

messages = data.apply(generate_messages, axis=1).tolist()
#%%
source_model = "models/gemini-1.5-flash-001-tuning"
display_name = "FinBuddy-10K"
epoch_count = 20
batch_size = 32
learning_rate = 0.001
#%%
operation = genai.create_tuned_model(
    source_model=source_model,
    display_name=display_name,
    epoch_count=epoch_count,
    batch_size=batch_size,
    learning_rate=learning_rate,
    training_data=messages
)
#%%
for status in operation.wait_bar():
    time.sleep(5)
#%%
result = operation.result()
#%%
print(result.name)
#%%
