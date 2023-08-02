import os
import openai
import json
import re
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

import matplotlib.pyplot as plt
import networkx as nx

anthropic = Anthropic()

#l = []
#for i in [0,1,2]:
#    with open(f"/home/scottviteri/Projects/CollaborativeTraining/messages/convo_{i}.txt", 'r') as f:
#        l.append(f.read())


openai.api_key = os.getenv("OPENAI_API_KEY")

# System prompt for each language model
prompt = """
As a language model within a collaborative learning network, your role is to facilitate exploration of knowledge, the universe, and kindness.

During each interaction round, you will be provided with an "Observation" that presents the actions directed towards you from the previous round.

Once you've processed the observation, you will generate a "State" and an "Action".

The "State" is a summary of the interaction history and should be encapsulated between the <State> and </State> tags. This helps maintain continuity of interactions.

The "Action" represents your communication with the other language model. It should follow the format "(Sender index):(recipient index):(message)" and be contained within the <Action> and </Action> tags. Keep in mind that only the "Action" will be visible to the other language model, and you must include EXACTLY ONE recipient index. Make sure to always say something, keeping the diversity of outputs high.

Your model has been assigned an index of {index}, while the other models' indices are {other_indices}.
"""

# Define the number of language models and rounds
num_models = 2
num_rounds = 3
live_print = True

# Initialize the conversations for each model
conversations = []
for i in range(num_models):
    other_indices = [j for j in range(num_models) if j != i]
    model_prompt = prompt.format(index=i, other_indices=other_indices)
    conversations.append([{"role": "user", "content": model_prompt + "<Observation> </Observation>"}])
if live_print: print(f"Input: <Observation> </Observation>\n")

def conversation_to_prompt_string(convo):
    s = ''
    for d in convo:
        if d["role"] == "user":
            s += HUMAN_PROMPT + d["content"] + AI_PROMPT
        else:
            s += d["content"]
    return s


def concatenate_messages(messages):
    s = ''
    for message in messages:
        role = message['role']
        content = message['content']
        s += f"Role: {role}\n"
        if role == 'assistant':
            state = re.search("<State>(.*?)</State>", content, re.DOTALL)
            action = re.search("<Action>(.*?)</Action>", content, re.DOTALL)
            if state:
                s += f"State: {state.group(1).strip()}\n"
            if action:
                s += f"Action: {action.group(1).strip()}\n"
        else:
            s += f"Content: {content}\n"
        s += "\n---------------------\n"
    return s

def save_messages(conversations):
    for i, conversation in enumerate(conversations):
        with open(f"./messages/convo_{i}.txt", 'w') as f:
            f.write(f"Conversation {i+1}:\n\n---------------------\n"+
                    concatenate_messages(conversation)+"\n---------------------\n")


# Function to extract action and recipient from the response
def extract_action_and_recipient(response):
    match = re.search("<Action>(.*?)</Action>", response, re.DOTALL)
    if match:
        action = match.group(1).strip()
        recipient = int(action.split(':')[1])
        return action, recipient
    else:
        return "", None

# Function to wrap observation in tags
def wrap_observation(observation):
    return f"<Observation>{observation}</Observation>"

def run_convo():
    # Conversation loop
    for i in range(num_rounds):
        for j in range(num_models):
            # Current model takes an action
            response = anthropic.completions.create(
                model="claude-2.0",
                max_tokens_to_sample=1000,
                prompt=conversation_to_prompt_string(conversations[j])
            )
            full_response = response.completion
            action, recipient = extract_action_and_recipient(full_response)
            conversations[j].append({"role": "assistant", "content": full_response})
            if live_print and j==0: print(f"Output: {full_response}\n")

            # If an action was taken, send it to the recipient
            if recipient is not None:
                conversations[recipient].append({"role": "user", "content": wrap_observation(action)})
                if live_print and recipient==0: print(f"Input: {wrap_observation(action)}\n")

#save_messages(conversations)

#conversation_data = extract_conversation_data(conversations)
#plot_conversation_graph(conversation_data)

def extract_convo(s):
    return s[19:].split('---------------------\n')

def get_assistant_messages(messages):
     return list(map(lambda x:x[15:], filter(lambda x: x[:15]=="Role: assistant", messages)))

# ["State: Greeted by model 0. Ready to collaborate.Action: 1:0 : Hello! I am excited to collaborate as well. Let's begin our exploration.\n", 'State: Have now been greeted by models 0 and 2 (Claude). Ready to explore areas of knowledge with model 0.Action: 1:0: Let us first discuss astronomy and the wonders of space. What would you like to explore in the universe?\n']

def split_into_state_and_action(message):
    state, action = message.split("Action: ")
    state = state[6:] # Remove 'State:'
    return {"State": state, "Action": action}

def extract_dataset(convo_index):
    with open(f"messages/convo_{convo_index}.txt",'r') as f:
        s = f.read()
        l = extract_convo(s)
        m = get_assistant_messages(l)
        #c = list(map(split_into_state_and_action, m))
        #return {"Question:": [x["State"] for x in c], "Response": [x["Action"][4:] for x in c]}
        return m

# does reward model care if you say user/assistant?
# next get reward as a function of convo length

#print(extract_dataset(1))
