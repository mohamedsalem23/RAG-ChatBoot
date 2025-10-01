# 🤖 AI Faculty Chatbot - Kafr Elsheikh University

This project is a **chatbot application** for the Faculty of Artificial Intelligence at Kafr Elsheikh University.  
It is built using **Streamlit**, **LangChain**, and **Google Generative AI (Gemini)**.  
The bot answers questions related to university policies, study schedules, and organized data from uploaded PDFs.

---

## 🚀 Features
- 📄 Load and process multiple PDF documents.  
- 🔎 Split text into chunks and store them in a vector database (Chroma).  
- 🧠 Use HuggingFace multilingual embeddings (`intfloat/multilingual-e5-large`).  
- 💬 Chat with Google Gemini (`gemini-2.5-flash`) for context-based Q&A.  
- 🗂️ Automatic table extraction: if the response is a Markdown table, it is displayed as a formatted DataFrame.  
- 🌍 Supports Arabic (RTL) UI for better readability.  

---

## 🛠️ Installation



