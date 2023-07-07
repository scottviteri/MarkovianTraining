def extract_convo(s):
    return s[19:].split('---------------------\n')

def get_assistant_messages(messages):
     return list(map(lambda x:x[15:], filter(lambda x: x[:15]=="Role: assistant", messages)))

# ["State: Greeted by model 0. Ready to collaborate.Action: 1:0 : Hello! I am excited to collaborate as well. Let's begin our exploration.\n", 'State: Have now been greeted by models 0 and 2 (Claude). Ready to explore areas of knowledge with model 0.Action: 1:0: Let us first discuss astronomy and the wonders of space. What would you like to explore in the universe?\n']

def split_into_state_and_action(message):
    state, action = message.split("Action: ")
    state = state[6:] # Remove 'State:'
    return {"State": state, "Action": action}

with open("messages/convo_1.txt",'r') as f:
    s = f.read()
    l = extract_convo(s)
    m = get_assistant_messages(l)
    c = list(map(split_into_state_and_action, m))
    dataset = {"Question:": [x["State"] for x in c], "Response": [x["Action"][4:] for x in c]}
    print(dataset)
