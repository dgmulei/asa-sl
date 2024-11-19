# Asa - Bainbridge Island Real Estate Advisor

A Streamlit-powered chatbot that provides real estate guidance using document-based knowledge and GPT-4.

## Features

- Interactive chat interface
- Document-based knowledge retrieval
- Real-time document processing
- Secure conversation handling
- Professional real estate guidance

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd asa-sl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=all-MiniLM-L6-v2
CHROMA_DB_PATH=./chroma_db
DOCUMENTS_PATH=./data/real_estate_docs
```

4. Run locally:
```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add the following secrets in Streamlit Cloud settings:
   - OPENAI_API_KEY

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
├── README.md
└── requirements.txt
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_NAME`: The name of the sentence transformer model (default: all-MiniLM-L6-v2)
- `CHROMA_DB_PATH`: Path to store the ChromaDB database
- `DOCUMENTS_PATH`: Path to the documents directory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
