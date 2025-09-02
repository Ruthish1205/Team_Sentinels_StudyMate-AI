# 📚 StudyMate – AI-Powered Academic Assistant

StudyMate is an AI-powered academic assistant that allows students and researchers to **interact with their study materials** (PDFs like textbooks, lecture notes, research papers) in a **conversational, question-answering format**.

Instead of scrolling through long documents or manually searching, just upload your PDFs and ask natural language questions — StudyMate will return contextual, reference-based answers using the power of AI.

---

## 🚀 Features

- ✅ Upload one or more PDF files
- ✅ Extracts and indexes content using vector embeddings
- ✅ Ask natural language questions and get answers from your PDFs
- ✅ Powered by OpenAI GPT and FAISS for accurate semantic search
- ✅ CLI-based — light and easy to use

---

## 🧰 Tech Stack

- 🧠 **OpenAI GPT (via API)**
- 📄 **PyPDF2** – PDF text extraction
- 🔍 **FAISS** – Vector similarity search
- 🔗 **LangChain** – Document management & LLM integration
- 🐍 Python 3.8+

---

## 🛠 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/studymate.git
cd studymate
2. Install Dependencies
bash
Copy code
pip install -r requirements.txt
If you don’t have a requirements.txt, create one with:

bash
Copy code
pip install PyPDF2 openai langchain faiss-cpu tiktoken
pip freeze > requirements.txt
3. Set OpenAI API Key
Create an environment variable for your API key:

bash
Copy code
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
⚠️ Never share your API key publicly!

💡 Usage
bash
Copy code
python studymate.py
Example
bash
Copy code
🔍 Welcome to StudyMate!
Enter PDF file paths (comma-separated): sample_notes.pdf, textbook.pdf

✅ PDFs loaded and text extracted.
📚 Splitting and indexing documents...
✅ Ready to answer your questions!

Ask a question (or type 'exit' to quit): What is Newton's second law?

🧠 Answer: Newton's second law states that the force applied on an object is equal to its mass times acceleration...
📁 File Structure
bash
Copy code
studymate/
│
├── studymate.py           # Main CLI script
├── requirements.txt       # Python dependencies
└── README.md              # You're here
🌟 Future Improvements
 Web interface using Streamlit or Gradio

 Source referencing with page numbers

 Upload drag-and-drop support

 Save/load vector store for persistent sessions

🤝 Contributing
Contributions are welcome! Feel free to open an issue or pull request if you want to:

Add new features

Fix bugs

Improve code structure or documentation

📄 License
This project is licensed under the MIT License.

🙋‍♀️ Author
Your Name – Ruthish Kumar S 

⭐️ Star the Repo
If you found this project helpful, consider starring the repo 🌟 to support the development.


