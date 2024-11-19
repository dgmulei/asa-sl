# Asa - Bainbridge Island Real Estate Advisor

A Streamlit-powered chatbot that provides real estate guidance using document-based knowledge and GPT-4.

## Features

- Interactive chat interface
- Document-based knowledge retrieval
- Real-time document processing
- Secure conversation handling
- Professional real estate guidance

## System Requirements

- Python 3.10
- SQLite3
- Build essentials

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd asa-sl
```

2. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev sqlite3 libsqlite3-dev python3-pip
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=all-MiniLM-L6-v2
CHROMA_DB_PATH=./chroma_db
DOCUMENTS_PATH=./data/real_estate_docs
```

5. Run locally:
```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Connect your GitHub repository

4. Configure the deployment:
   - Main file path: app.py
   - Python version: 3.10 (specified in runtime.txt)
   - Required environment variables:
     - OPENAI_API_KEY
     - MODEL_NAME
     - CHROMA_DB_PATH
     - DOCUMENTS_PATH

5. The following files handle deployment configuration:
   - `requirements.txt`: Python package dependencies
   - `packages.txt`: System-level dependencies
   - `runtime.txt`: Python version specification
   - `.streamlit/config.toml`: Streamlit configuration

## Project Structure

```
.
├── .streamlit/
│   └── config.toml
├── data/
│   └── real_estate_docs/
├── utils/
│   ├── __init__.py
│   ├── conversation_manager.py
│   ├── document_loader.py
│   ├── embeddings_manager.py
│   └── query_engine.py
├── .env
├── .gitignore
├── app.py
├── packages.txt
├── requirements.txt
├── runtime.txt
└── README.md
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_NAME`: The name of the sentence transformer model (default: all-MiniLM-L6-v2)
- `CHROMA_DB_PATH`: Path to store the ChromaDB database
- `DOCUMENTS_PATH`: Path to the documents directory

## Troubleshooting

If you encounter deployment issues:
1. Ensure all system dependencies are available (specified in packages.txt)
2. Verify Python version compatibility (3.10)
3. Check that all environment variables are properly set in Streamlit Cloud
4. Verify ChromaDB has proper write permissions in the specified path

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
