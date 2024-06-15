import json
import os
import re
from openai import OpenAI
from tqdm import tqdm

src_json = 'mmbench.jsonl'
# my_api_key = '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'
my_api_key = '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'
client = OpenAI(api_key=my_api_key)

patterns = [r'\[([A-Za-z])\]', r'^([a-zA-Z])\s']

answer_dats = []
with open(src_json, 'r') as f:
    for line in f:
        answer_dats.append(json.loads(line))

count = 0
for dat in tqdm(answer_dats):
    output = dat['output']
    gt = dat['gt']
    question = dat['input']
    choices = {}
    for key in ['A', 'B', 'C', 'D']:
        pattern = re.compile('[{}](.*?)(?=\n|$)'.format(key))
        content = re.search(pattern, question)
        if content:
            choices[key] = content.group(0)
    options = ''
    for key in choices.keys():
        options += choices[key]
    options = options.strip()

    output_choice = None
    if output in ['a', 'b', 'c', 'd', 'A', 'B', 'C', 'D']:
        output_choice = output.upper()
    else:
        for pattern in patterns:
            res = re.search(pattern, output)
            if res:
                # print(output)
                output_choice = res.group(0).upper()
                break
    
    if output_choice is None:
        # print('Options:', options)
        # print('Answer: ', output)
        gpt_query_template = (
        "You are an AI assistant to help me matching an answer with several options of a multiple choice question. "
        "You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output X. "\
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and X. \n"
        "Example 1: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: a cute teddy bear\nYour output: A\n"
        "Example 2: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: Spider\nYour output: X\n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: the correct answer is d\nYour output: D\n"
        "Example 3: \n"
        f"Question: {question}?\nOptions: {options}\nAnswer: {output}\nYour output: ")

        messages = [{"role": "system", "content": gpt_query_template}]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
        )
        question_response = response.choices[0].message.content
        # print('Response:', question_response)
        for pattern in patterns:
            res = re.search(r'^([a-zA-Z])\s?', question_response)
            if res:
                output_choice = res.group(0).upper()

        # print('Choice:', output_choice)

    if output_choice == gt:
        count += 1

print('acc: {}'.format(count / len(answer_dats)))