<p align="center">
  <img src="https://github.com/user-attachments/assets/db3d47be-4bb8-4a87-82b7-1b8cfd8afe56" alt="PulseQuery AI Logo" width="100"/>
</p>

<h1 align="center">PulseQuery AI</h1>

---

## 🩺 Advanced Medical AI with MedGemma, RAG, & Document Processing

**PulseQuery AI** is an advanced, privacy-focused medical AI system designed to deliver accurate, evidence-backed answers to clinical queries. Built for deployment within hospital private networks, it combines **state-of-the-art medical AI models**, **Retrieval-Augmented Generation (RAG)**, and **secure document processing**.

---

## 📋 Overview

PulseQuery AI is built around three core principles:
1. **Accuracy** – Powered by a specialized medical language model (MedGemma 4B).
2. **Contextual Understanding** – Enhanced with RAG and medical embeddings for grounded responses.
3. **Security & Privacy** – All processing is performed locally, with role-based access control.

### Core Components
- **MedGemma 4B** – Optimized for medical queries with GPU acceleration.
- **MedEmbed** – Specialized medical embedding models for semantic search and retrieval.
- **RAG System** – Retrieval-Augmented Generation to ensure responses are fact-based and grounded in your own knowledge base.
- **Document Processing** – Supports ingestion of PDFs, DOCX, and TXT files.
- **Role-Based Authentication** – Different access levels for doctors, nurses, admins, and residents.

---

## ✨ Key Features

- **🤖 AI-Powered Medical Reports** – Generate domain-specific, accurate medical answers.
- **📄 Secure Document Upload & Processing** – Ingest and process local medical documents with no external API calls.
- **🔍 Semantic Search** – Medical-specialized retrieval to surface the most relevant content.
- **🧠 RAG-Enhanced Responses** – Reduces hallucinations by grounding outputs in real data.
- **🔐 Role-Based Permissions** – Fine-grained control over access rights.
- **🖥️ Web Interface** – Simple UI for testing and interaction.
- **⚡ GPU Support** – CUDA acceleration for high-performance inference.

---

## 🛠️ Prerequisites

### System Requirements
- **Python**: 3.8+
- **RAM**: 8GB minimum (16GB recommended for GPU workloads)
- **Storage**: 10GB+
- **GPU** (optional): CUDA-compatible NVIDIA GPU for acceleration

### Dependencies
Key packages:
```bash
Flask>=2.0
torch>=1.9
transformers>=4.20
sentence-transformers
chromadb
langchain-community
llama-cpp-python
pypdf
python-docx
```

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/pulsequery-ai.git
cd pulsequery-ai
```

### **2. Create Virtual Environment**
# Windows
```bash
python -m venv venv
venv\Scripts\activate
```

# Linux/Mac
```bash
python -m venv venv
source venv/bin/activate
```
### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```
# Or install manually:

```bash
pip install flask torch transformers sentence-transformers
pip install chromadb langchain-community llama-cpp-python pypdf python-docx
```

### **4. Download Models**
Create models/ directory and download medgemma-4b-it-Q8_0.gguf.

```bash
mkdir models
```
# Place medgemma-4b-it-Q8_0.gguf in models/

```text
pulsequery-ai/
├── app.py
├── config/
│   └── config.py
├── core/
│   ├── medgemma_inference.py
│   └── rag_system.py
├── services/
│   ├── auth_service.py
│   ├── session_manager.py
│   └── auth_decorators.py
├── models/
│   └── medgemma-4b-it-Q8_0.gguf
├── data/
│   └── chromadb/
├── tests/
│   └── test_rag.py
├── logs/
│   └── app_YYYYMMDD_HHMMSS.log
└── requirements.txt
```

### ** Configuration**
Environment Variables

Create .env file:
```shell
MEDGEMMA_MODEL_PATH=models/medgemma-4b-it-Q8_0.gguf
RAG_DATA_DIR=data/chromadb
FLASK_ENV=development
FLASK_DEBUG=True
```
### **Config File (config/config.py)**
```python
class Config:
    SECRET_KEY = 'change-me-in-production'
    HOST = '127.0.0.1'
    PORT = 5000
```
### **Step 5 – Running the Application**
Start the Flask application:
`python app.py`
### *You should see:*
✅ MedGemma import successful
✅ RAG initialized
🌐 Server running at http://localhost:5000

## 🌐 Step 6 – Access the Interface

- **Main Server**: [http://localhost:5000](http://localhost:5000)  
- **Test UI**: [http://localhost:5000/test-ui](http://localhost:5000/test-ui)  
- **Health Check**: [http://localhost:5000/health](http://localhost:5000/health)  

### Demo Credentials

| Role     | Username    | Password     |
|----------|-------------|--------------|
| Doctor   | doctor1     | password123  |
| Admin    | admin1      | admin123     |
| Nurse    | nurse1      | nurse123     |
| Resident | resident1   | resident123  |


### **🧪 Step 7 – Testing**
python tests/test_medembed.py
### **Test RAG**
python tests/test_rag.py


### **Step 8 – API Endpoints**
## Authentication
POST /api/auth/login

### Document Upload
POST /api/rag/upload


### Semantic Search
POST /api/rag/search

### AI Generation
POST /api/medgemma/generate-with-rag


## Step 9 – Key Improvements Over Original Plan

1. **N-gram → Transformer Embeddings** – Better semantic understanding.  
2. **Statistical Fluency → Medical Prompt Templates** – Domain-specific prompts.  
3. **Basic Tokenization → Medical Info Extraction** – Structured patient data.  
4. **Simple Scoring → Multi-Dimensional Metrics** – Context, terminology density, etc.  

###  Step 10 – Troubleshooting
RAG Not Available
python tests/test_rag.py
### Model Not Loading
## Verify path to .gguf model file.
### Ensure 8GB+ RAM or force CPU mode:
bash
set CUDA_VISIBLE_DEVICES=
### Import Errors
bash
pip install --upgrade langchain-community

## Step 11 – Performance Optimization

- **Enable GPU acceleration** if available.  
- **Use ChromaDB** for fast vector search.  
- **Process documents in chunks** for memory efficiency.  

## Step 12 – Security Best Practices
- **Change default credentials** before production.  
- **Use HTTPS** for secure communication.  
- **Store all medical data locally** to ensure privacy.  

---

## Step 13 – Monitoring
- **System Health** – `/health` endpoint.  
- **Logs** – stored in `/logs`.  

---

## Step 14 – Contributing
1. Fork repository  
2. Create a feature branch  
3. Commit & push changes  
4. Submit a Pull Request  

---

## Step 15 – License
MIT License – see `LICENSE`.  

---

## Step 16 – Support
If you face issues:  
- Check `/logs`  
- Run test scripts  
- Open GitHub issue with:  
  - Error logs  
  - System configuration  
  - Steps to reproduce  


### *PulseQuery AI – Advancing Medical AI Technology with Trust & Accuracy*

---

> 💡 Do you want me to also **add GitHub badges** and a **system architecture diagram section** so this README looks more like a professional open-source project?  
> This would make it visually stand out and more appealing to contributors and users.





