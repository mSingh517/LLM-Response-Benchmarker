import json
import re
import pandas as pd


def get_text(filename):
    with open(filename, 'r') as file:
        data = file.read()

    return [x for x in data.split('==') if x]


def parse_json(llm_text):
    try:
        stripped = re.sub('```', '', str(llm_text)).strip()
        return json.loads(stripped)
    except Exception as e:
        return f'JSON parsing error: {str(e)}\nLLM Output:\n{llm_text}'


def get_all_data():
    all_data = [[None for _ in range(4)] for _ in range(6)]
    files = [
        'gpt.txt',
        'claude.txt',
        'gemini.txt',
        'mistral.txt'
    ]

    for j, filename in enumerate(files):
        curr_txt = get_text(filename)
        for i, data in enumerate(curr_txt):
            curr_json = parse_json(data)
            all_data[i][j] = curr_json

    return pd.DataFrame(all_data, columns=['ChatGPT', 'Claude', 'Gemini', 'Mistral'])


if __name__ == '__main__':
    out = get_all_data()
    out.to_pickle('all_data_df.pkl')
    loaded_df = pd.read_pickle('all_data_df.pkl')
    print(loaded_df)
