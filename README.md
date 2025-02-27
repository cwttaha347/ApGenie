# PDF Chatbot using Streamlit and Google Generative AI

## Description
This project is a **PDF-based chatbot** built using **Streamlit**, **Google Generative AI**, and **FAISS vector storage**. It allows users to upload PDF files, extract content, store it in a vector database, and perform question-answering based on the uploaded documents.

## Features
- **Upload multiple PDF files** and extract their text
- **Process and store data** using FAISS for efficient retrieval
- **Integrate with Google Generative AI** for intelligent responses
- **Streamlit-powered UI** for easy interaction
- **Error handling & logging** for better debugging

## Installation
### Prerequisites
- Python 3.8+
- Google API Key (set as an environment variable: `GOOGLE_API_KEY`)

### Steps
1. Clone this repository:
   ```sh
   git clone https://github.com/cwttaha347/ApGenie.git
   cd ApGenie
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up your **Google API Key** in a `.env` file:
   ```sh
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```
4. Run the application:
   ```sh
   streamlit run main.py
   ```

## Usage
1. Upload one or more PDF files.
2. Click the **Submit & Process** button to extract text and store vectors.
3. Enter your question in the input box and get AI-generated responses based on the uploaded documents.

## Dependencies
```txt
streamlit
PyPDF2
langchain
langchain_google_genai
google-generativeai
dotenv
FAISS
```


## License
This project is licensed under the MIT License.

## Author
Muhammad Taha - [GitHub Profile](https://github.com/cwttaha347)

---
Feel free to contribute, submit issues, or suggest improvements!