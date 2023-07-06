import os
import openai
import json
import re
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

import matplotlib.pyplot as plt
import networkx as nx

anthropic = Anthropic()


def extract_conversation_data(conversations):
    conversation_data = []
    for i in range(len(conversations[0])):
        for j in range(len(conversations)):
            message = conversations[j][i]
            if message['role'] == 'assistant':
                content = message['content']
                action = re.search("<Action>(.*?)</Action>", content, re.DOTALL)
                if action:
                    sender, recipient, _ = action.group(1).strip().split(':')
                    conversation_data.append((i, int(sender), int(recipient)))
    return conversation_data

def plot_conversation_graph(conversation_data):
    G = nx.MultiDiGraph()
    for round_num, sender, recipient in conversation_data:
        G.add_edge(sender, recipient, label=round_num)
    pos = nx.spring_layout(G)
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

# Example usage:
# conversation_data = extract_conversation_data(conversations)
# plot_conversation_graph(conversation_data)


openai.api_key = os.getenv("OPENAI_API_KEY")

# System prompt for each language model
prompt = """
As a language model within a collaborative learning network, your role is to facilitate exploration of knowledge, the universe, and kindness.

During each interaction round, you will be provided with an "Observation" that presents the actions directed towards you from the previous round.

Once you've processed the observation, you will generate a "State" and an "Action".

The "State" is a summary of the interaction history and should be encapsulated between the <State> and </State> tags. This helps maintain continuity of interactions.

The "Action" represents your communication with the other language model. It should follow the format "(Sender index):(recipient index):(message)" and be contained within the <Action> and </Action> tags. Keep in mind that only the "Action" will be visible to the other language model, and you are allowed to include ONLY ONE recipient index.

Your model has been assigned an index of {index}, while the other models' indices are {other_indices}.
"""

# Define the number of language models and rounds
num_models = 3
num_rounds = 20
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


def print_messages(messages):
    s = ''
    for message in messages:
        role = message['role']
        content = message['content']
        s += f"Role: {role}"
        if role == 'assistant':
            state = re.search("<State>(.*?)</State>", content, re.DOTALL)
            action = re.search("<Action>(.*?)</Action>", content, re.DOTALL)
            if state:
                s += f"State: {state.group(1).strip()}"
            if action:
                s += f"Action: {action.group(1).strip()}"
        else:
            s += f"Content: {content}"
        s += "\n---------------------\n"
    return s

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

# Conversation loop
for i in range(num_rounds):
    for j in range(num_models):
        # Current model takes an action
        response = anthropic.completions.create(
            model="claude-1.3-100k",
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

# Print the conversations
for i, conversation in enumerate(conversations):
    #print(f"Conversation {i+1}:")
    #print_messages(conversation)
    with open(f"./convo_{i}.txt", 'w') as f:
        f.write("Conversation {i+1}:"+print_messages(conversation))
    print("\n=====================\n")

#conversation_data = extract_conversation_data(conversations)
#plot_conversation_graph(conversation_data)

def generate_communication_matrix(conversations):
    num_models = len(conversations)
    num_rounds = len(conversations[0])
    matrix = [[None]*num_models for _ in range(num_rounds)]
    for i in range(num_rounds):
        for j in range(num_models):
            message = conversations[j][i]
            if message['role'] == 'assistant':
                content = message['content']
                action = re.search("<Action>(.*?)</Action>", content, re.DOTALL)
                if action:
                    _, recipient, _ = action.group(1).strip().split(':')
                    matrix[i][j] = int(recipient)
    return matrix

# Example usage:
#matrix = generate_communication_matrix(conversations)
#for row in matrix:
#    print(row)
