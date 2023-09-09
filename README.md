# CollaborativeTraining

## Installation
First git clone the repository.
Then, from within the repository and your env that has python3, run:
```
pip install -e .
```


For llama, you will have to install that separately at the moment.

## Code Structure
📦CollaborativeTraining
 ┣ 📂LlamaCodeSnippets
 ┃ ┣ 📜gpu_job.sh
 ┃ ┣ 📜llama-jupyter-notebook-cluster.png
 ┃ ┣ 📜llama7b-accelerate.py
 ┃ ┣ 📜llama7b-accelerate2.py
 ┃ ┗ 📜peter-mock-llama.py
 ┣ 📂Prompting
 ┃ ┣ 📜CooperativeTraining.png
 ┃ ┣ 📜kindergarden.png
 ┃ ┗ 📜simplified-loop.txt
 ┣ 📂data
 ┃ ┗ 📜st_patrick_biography.txt
 ┣ 📂results - aggregates the results
 ┣ 📂src
 ┃ ┣ 📂collaborative_experiments
 ┃ ┃ ┣ 📜constants.py
 ┃ ┃ ┣ 📜mvp_loss_decrease.py
 ┃ ┃ ┗ 📜test_mvp.ipynb
 ┣ 📂tests - file with all the pytests
 ┃ ┗ 📜mvp_test.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜convo_claude.py
 ┣ 📜convo_multi.py
 ┣ 📜convo_multi_claude.py
 ┣ 📜data.py
 ┣ 📜evaluate_llama_model.py
 ┣ 📜evaluate_llama_model_direct.py
 ┣ 📜requirements.txt
 ┗ 📜setup.py