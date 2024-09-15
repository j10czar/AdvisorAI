# AdvisorAI
A AI powered UF academic advisor
<br/>
<img width="721" alt="image" src="https://github.com/user-attachments/assets/c3895c53-4aec-47af-bd12-22240e0eab4c">

<br/>
Cannot afford AWS hosting so here is how to install locally :(


## How to run application locally
1. Clone repo and install dependencies

```
pip install -r requirements.txt
```

2. Install Ollama from this link
https://ollama.com/

3. Run the following commands

```
Ollama run mistral
```
```
Ollama run nomic-embed-text
```
<br/>
then pull the models...

```
ollama pull mistral
```


4. Run the application
```
streamlit run app.py
```

Enjoy <3


