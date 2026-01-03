import re
import json
import pandas as pd
from openai import OpenAI
import anthropic
from google import genai
from mistralai import Mistral
from llm_prompt_utils import llms, prompts


def openai_interaction(api_key, prompt):
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {str(e)}"


def claude_interaction(api_key, prompt):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Claude API error: {str(e)}"


def gemini_interaction(api_key, prompt):
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-1.5-flash', contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Gemini API error: {str(e)}"


def mistral_interaction(api_key, prompt):
    try:
        client = Mistral(api_key=api_key)
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return chat_response.choices[0].message.content
    except Exception as e:
        return f"Mistral API error: {str(e)}"


def parse_json(llm_text):
    try:
        stripped = re.sub('```', '', str(llm_text)).strip()
        return json.loads(stripped)
    except Exception as e:
        return f'JSON parsing error: {str(e)}\nLLM Output:\n{llm_text}'


def generate_configs_one_llm(model, api_key, llm_prompts):
    """Generate responses for each prompt using one LLM"""
    model_selector = {
        'ChatGPT': openai_interaction,
        'Claude': claude_interaction,
        'Gemini': gemini_interaction,
        'Mistral': mistral_interaction
    }
    configs = []
    for prompt in llm_prompts:
        model_output = model_selector[model](api_key, prompt)
        parsed = parse_json(model_output)
        configs.append(parsed)
    return configs


def generate_configs_one_prompt(api_keys, prompt):
    """Generate responses for one prompt using each LLM"""
    gpt_output = openai_interaction(api_keys[0], prompt)
    claude_output = claude_interaction(api_keys[1], prompt)
    gemini_output = gemini_interaction(api_keys[2], prompt)
    mistral_response = mistral_interaction(api_keys[3], prompt)

    configs = [
        parse_json(gpt_output),
        parse_json(claude_output),
        parse_json(gemini_output),
        parse_json(mistral_response)
    ]

    return configs


def get_all_results(api_keys):
    """
    Create Pandas DataFrame that stores all LLMs responses to all prompts.
    Rows correspond to prompts, columns to LLMs.

    Columns: ChatGPT, Claude, Gemini, Mistral
    """
    all_data = []
    for prompt in prompts:
        configs = generate_configs_one_prompt(api_keys, prompt)
        all_data.append(configs)
    return pd.DataFrame(all_data, columns=llms)
