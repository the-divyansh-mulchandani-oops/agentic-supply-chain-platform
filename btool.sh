python3 -m venv .venv
source .venv/bin/activate
pip install -U -r requirements.txt
cp .env.example .env
ollama pull llama3.2:1b
python3 chatbot.py
