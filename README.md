⚠️ Important Download medgemma-4b-it-Q8_0 from https://huggingface.co/unsloth/medgemma-4b-it-GGUF/tree/main and place it in folder models.

https://miro.com/app/board/uXjVJYWSi6c=/?share_link_id=134124900492 Refer Mind Map for Start



-----

# 🏥 PulseQuery AI - Medical AI System

### **Advanced Medical AI System with MedGemma Model, Medical Document Processing & RAG Technology**

-----

## 📋 Overview

**PulseQuery AI** is a comprehensive medical AI system designed to provide accurate, evidence-backed answers to clinical queries. It combines cutting-edge AI technologies with a strong emphasis on data privacy and security, making it suitable for use within a hospital's private network.

The system's core components include:

  * **MedGemma 4B**: A specialized medical language model with GPU acceleration.
  * **MedEmbed**: A family of specialized embedding models for medical and clinical data, used for document retrieval and semantic search.
  * **RAG System**: Retrieval-Augmented Generation technology that grounds AI responses in a local knowledge base.
  * **Document Processing**: Ingestion and analysis of medical documents in formats like PDF, DOCX, and TXT.
  * **Authentication**: Role-based access control for different medical professionals.

## 🎯 Features

  * **🤖 Medical AI Generation**: Provides specialized medical report generation and answers.
  * **📄 Document Upload & Processing**: Allows for the ingestion and analysis of local medical documents.
  * **🔍 Semantic Search**: Enables medical-specialized document retrieval to find highly relevant information.
  * **🧠 RAG-Enhanced Responses**: Ensures AI responses are context-aware and grounded in a trusted knowledge base, reducing the risk of "hallucinations."
  * **🔐 Authentication System**: Implements role-based permissions for doctors, nurses, and administrators.
  * **🖥️ Web Interface**: A complete user interface for easy interaction and system testing.
  * **⚡ GPU Support**: Includes CUDA acceleration for faster model inference and processing.

## 🛠️ Prerequisites

### **System Requirements**

  * **Python**: 3.8 or higher
  * **RAM**: 8GB+ (16GB recommended for GPU usage)
  * **Storage**: 10GB+ free space for models and data
  * **GPU** (Optional): CUDA-compatible for acceleration (e.g., an NVIDIA card)

### **Dependencies**

  * `Flask` 2.0+
  * `PyTorch` 1.9+
  * `Transformers` 4.20+
  * `LangChain Community`
  * `ChromaDB`
  * `Sentence Transformers`
  * `llama-cpp-python`
  * `pypdf`
  * `python-docx`
  * And others as listed in `requirements.txt`

## 🚀 Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/pulsequery-ai.git
cd pulsequery-ai
```

### **2. Create Virtual Environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**

```bash
# Core dependencies
pip install -r requirements.txt

# Or install manually:
pip install flask torch transformers sentence-transformers
pip install chromadb langchain-community
pip install llama-cpp-python pypdf python-docx
pip install sqlite3 uuid typing-extensions
```

### **4. Download Models**

Create a `models/` directory and download the MedGemma model. This project uses the `medgemma-4b-it-Q8_0.gguf` file format for efficient, local inference with `llama-cpp-python`.

```bash
mkdir models
# Download medgemma-4b-it-Q8_0.gguf to models/ directory
# (Replace with your actual model download method)
```

### **5. Project Structure Setup**

Ensure your project structure looks like this:

```text
pulsequery-ai/
├── app.py                 # Main Flask application
├── config/
│   └── config.py         # Configuration settings
├── core/
│   ├── medgemma_inference.py  # MedGemma model handler
│   └── rag_system.py         # RAG system implementation
├── services/
│   ├── auth_service.py       # Authentication service
│   ├── session_manager.py    # Session management
│   └── auth_decorators.py    # Auth decorators
├── models/
│   └── medgemma-4b-it-Q8_0.gguf  # Your model file
├── data/
│   └── chromadb/            # Vector database storage
├── tests/
│   └── test_rag.py         # Test scripts
├── logs/
│   └── app_20250805_084856.log         # Logs Store
└── requirements.txt

```

## ⚙️ Configuration

### **Environment Variables (Optional)**

Create a `.env` file for sensitive data and development settings:

```bash
MEDGEMMA_MODEL_PATH=models/medgemma-4b-it-Q8_0.gguf
RAG_DATA_DIR=data/chromadb
FLASK_ENV=development
FLASK_DEBUG=True
```

### **Config Settings**

Update `config/config.py` if needed:

```python
class Config:
    SECRET_KEY = 'your-secret-key-change-in-production'
    HOST = '127.0.0.1'
    PORT = 5000
    # Add other configuration as needed
```

## 🏃‍♂️ Running the Application

### **1. Start the Flask Server**

```bash
python app.py
```

### **2. Expected Startup Output**

```text
🔇 Verbose logging suppressed - console output cleaned
✅ MedGemma import successful
✅ RAG import successful
🔄 Creating enhanced RAG system instance...
🧠 Selected embedding model: abhinand/MedEmbed-base-v0.1
✅ RAG system initialized with medical embeddings
✅ Complete application created successfully!
🌐 Server running at http://localhost:5000
```

### **3. Access Points**

  * **Main Server**: `http://localhost:5000`
  * **Test UI**: `http://localhost:5000/test-ui`
  * **Health Check**: `http://localhost:5000/health`
  * **API Documentation**: See API Endpoints section below

## 🧪 Testing

### **Test Medical Embeddings**

```bash
python tests/test_medembed.py
```

**Expected output:**

```text
🧪 Testing MedEmbed model...
✅ MedEmbed model loaded successfully
✅ Embeddings computed successfully!
📊 Shape: (3, 768)
```

### **Test RAG System**

```bash
python tests/test_rag.py
```

**Expected output:**

```text
🧪 Testing RAG system directly...
✅ RAG system imported
🔄 Initializing RAG with medical embeddings...
✅ RAG system initialized successfully
✅ RAG system test completed successfully
```

## 🌐 Using the Web Interface

### **1. Open Test UI**

Navigate to: `http://localhost:5000/test-ui`

### **2. Login with Demo Credentials**

  * **Doctor**: `doctor1` / `password123` (Full access + upload)
  * **Admin**: `admin1` / `admin123` (Administrative access)
  * **Nurse**: `nurse1` / `nurse123` (Read-only access)
  * **Resident**: `resident1` / `resident123` (Limited access)

### **3. Test System Components**

  * **System Health**: Check all components' status.
  * **Model Status**: Verify MedGemma model is loaded.
  * **RAG Stats**: Check document database status.
  * **Document Upload**: Upload medical documents (requires write permission).
  * **AI Generation**: Test medical report generation.

## 📚 API Endpoints

### **Authentication**

```text
POST /api/auth/login
Content-Type: application/json

{
    "user_id": "doctor1",
    "password": "password123"
}
```

### **Document Management**

```text
# Upload Document
POST /api/rag/upload
X-Session-ID: your-session-id
Content-Type: multipart/form-data

# Search Documents
POST /api/rag/search
X-Session-ID: your-session-id
Content-Type: application/json
{
    "query": "diabetes management",
    "max_docs": 5
}
```

### **AI Generation**

```text
# Simple Generation
POST /api/medgemma/generate-simple
X-Session-ID: your-session-id
Content-Type: application/json
{
    "prompt": "Patient Name: John Doe\nChief Complaint: Chest pain"
}

# RAG-Enhanced Generation
POST /api/medgemma/generate-with-rag
X-Session-ID: your-session-id
Content-Type: application/json
{
    "prompt": "Patient Name: John Doe\nChief Complaint: Chest pain",
    "max_tokens": 600,
    "temperature": 0.3,
    "use_rag": true
}
```

### **System Status**

```text
GET /health                 # Detailed health check
GET /api/medgemma/status    # Model status
GET /api/rag/stats          # RAG system statistics
```

## 🔧 Troubleshooting

### **Common Issues & Solutions**

1.  **"RAG system not available" Error**

      * **Problem:** Document upload fails with a RAG unavailable error.
      * **Solutions:**
          * Check RAG system initialization: `python tests/test_rag.py`
          * Verify directories exist and are writable: `ls -la data/chromadb/`
          * Check for missing dependencies: `pip install chromadb sentence-transformers langchain-community`

2.  **Model Loading Fails**

      * **Problem:** MedGemma model not loading or app crashes on startup.
      * **Solutions:**
          * Verify model file exists: `ls -la models/medgemma-4b-it-Q8_0.gguf`
          * Check available memory: Ensure you have 8GB+ RAM available.
          * Try CPU-only mode if GPU fails: Set `CUDA_VISIBLE_DEVICES=""` to force CPU.

3.  **Import Errors**

      * **Problem:** `LangChain` deprecation warnings or import failures.
      * **Solutions:**
          * Update to latest versions: `pip install --upgrade langchain-community`
          * Or use specific compatible versions: `pip install langchain-community==0.0.20`

4.  **Permission Errors**

      * **Problem:** Cannot write to data directories or upload files.
      * **Solutions:**
          * Fix directory permissions: `chmod -R 755 data/`
          * Check disk space: `df -h`

5.  **"Control Token" Spam Messages**

      * **Problem:** Console flooded with tokenizer messages.
      * **Solution:** The app.py already includes noise filtering. If you're still seeing messages, verify the logging configuration is properly loaded.

### **Log Files**

The application creates log files for debugging in the `log/` directory.

```bash
# View latest log file
ls -la log/
tail -f log/app_*.log

# Search for specific errors
grep "❌" log/app_*.log
grep "RAG" log/app_*.log
```

## 📊 Performance Optimization

  * **GPU Acceleration**: Verify GPU availability with `python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"`
  * **Memory Management**: The system uses background loading to prevent the app from blocking and processes documents in chunks.
  * **Vector Storage**: `ChromaDB` is used for efficient similarity search.

## 🔒 Security Notes

### **Production Deployment**

  * Change default secret keys in `config.py`.
  * Use environment variables for sensitive data.
  * Enable HTTPS for production.
  * Review and update user credentials.
  * Implement proper session timeout.

### **Data Privacy**

  * Medical documents are stored locally.
  * No data is sent to external services.
  * Embeddings are computed locally.
  * User sessions are managed securely.

## 📈 Monitoring

  * **System Health Monitoring**: The `/health` endpoint provides component status.
  * **Log Files**: Detailed logging with timestamps.
  * **Memory Usage**: Monitor RAM usage during model loading.
  * **Storage**: Check available disk space.

## 🤝 Contributing

  * Fork the repository.
  * Create a feature branch: `git checkout -b feature-name`
  * Make changes and test thoroughly.
  * Commit changes: `git commit -am 'Add feature'`
  * Push to branch: `git push origin feature-name`
  * Create a Pull Request.

## 📄 License

This project is licensed under the MIT License - see `LICENSE` file for details.

## 🆘 Support

If you encounter issues:

  * Check the logs.
  * Run the individual test scripts.
  * Verify your setup.
  * Confirm adequate resources.
  * Review your configuration.

For additional support, create an issue in the repository with:

  * Full error message and stack trace.
  * Your system configuration.
  * Steps to reproduce the issue.
  * Log file contents (if applicable).

-----

### **🏥 PulseQuery AI - Advancing Medical AI Technology**
