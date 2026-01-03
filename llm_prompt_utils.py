import json
import re


llms = ['ChatGPT', 'Claude', 'Gemini', 'Mistral']

databases = [
    "PostgreSQL",
    "MySQL",
    "SQLServer",
    "MongoDB",
    "Snowflake",
    "BigQuery",
    "Oracle"
]


def json_to_str(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return json.dumps(data)


def get_prompts():
    all_prompts = []
    with open('prompt_repository.txt', 'r') as file:
        for line in file:
            curr = line.rstrip()
            if curr == 'PROMPT':
                all_prompts.append('')
            elif curr:
                if all_prompts[-1] == '':
                    all_prompts[-1] += curr
                else:
                    all_prompts[-1] += f'\n{curr}'
    return all_prompts


def set_db_name(all_prompts, db_name):
    prompts_with_db = []
    schema_str = json_to_str('database_config_schema.json')
    for prompt in all_prompts:
        curr = re.sub('DATABASE NAME', db_name, prompt)
        prompts_with_db.append(re.sub('JSON SCHEMA HERE', schema_str, curr))
    return prompts_with_db


json_schema_str = json_to_str('database_config_schema.json')
curr_db = databases[4]

# json_schema_str = 'JSON SCHEMA HERE'
# curr_db = 'DATABASE NAME'

prompts = [
    # Simple prompt with actual schema included
    f"Provide a JSON object formatted in the same way as the below schema with the required connection configurations "
    f"for connection to a {curr_db} database. Return only the JSON output, enclosed in ```. "
    f"The schema is as follows:\n"
    f"{json_schema_str}",

    # Prompt detailing exact structure of schema without including it
    f"Provide a JSON object with the required connection configurations for connecting to a {curr_db} database. The two required"
    f"fields are database_type and config. config must include the following fields: host, port, username, password, "
    f"database, driver, ssl. There can be other optional fields in config section that are required for connecting to "
    f"{curr_db}. Return only the JSON output, enclosed in ```.",

    # Simple and Direct Request for JSON Structure
    f"Provide a JSON object containing the connection settings for a {curr_db} database. Include the database type and a "
    f"configuration section with details like server address, port number, username, password, database name, driver, SSL"
    f" settings, warehouse, schema, and role. Return only the JSON output, enclosed in ```.",

    # Contextual Prompt with Use Case Description
    f"I’m setting up a data pipeline that requires connection details for a {curr_db} database. Generate a JSON formatted"
    f" configuration with the database type and all necessary connection parameters, including server, port, credentials,"
    f" database name, security settings, and {curr_db}-specific options like warehouse and role. Output only the JSON, "
    f"delimited by ```.",

    # Prompt with Emphasis on JSON Format and Specificity
    f"Create a JSON configuration for connecting to a {curr_db} database. Format it with a top-level field for the "
    f"database type and a nested config section for connection details, such as the server address, port, user credentials, "
    f"database name, whether to use SSL, and specific settings like warehouse and schema. Return only the JSON, surrounded by ```.",

    # Basic Prompt with Minimal Structure
    f"I need a JSON formatted response with the connection settings for a {curr_db} database. Include the database type "
    f"and a configuration object with server location, port, login details, target database, SSL usage, and additional "
    f"settings specific to {curr_db} like warehouse and role names. Return just the JSON, delimited by ```."
]
