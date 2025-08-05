import logging
import warnings
from datetime import datetime
import os

# ✅ ENHANCED: File + Console Logging Configuration
LOG_DIR = 'log'
LOG_FILENAME = f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
# Create the log directory if it does not exist
os.makedirs(LOG_DIR, exist_ok=True)
# Create the full log file path
log_file_path = os.path.join(LOG_DIR, LOG_FILENAME)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),  # Write to file
        logging.StreamHandler()  # Also display on console
    ]
)

# ✅ SUPPRESS: Tokenizer and model warnings (still needed)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('llama_cpp').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)

# ✅ CUSTOM FILTER: Block specific noisy messages
class NoiseFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if any(phrase in msg for phrase in [
            "control token",
            "is not marked as EOG", 
            "unused",
            "text generation"
        ]):
            return False
        return True

# Apply filter to root logger
logging.getLogger().addFilter(NoiseFilter())

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Create logger instance
logger = logging.getLogger(__name__)
logger.info(f"🗂️ Logging initialized - File: {LOG_FILENAME}")


from flask import Flask, jsonify, render_template_string, request, session
import os
import tempfile
import time
import uuid
import gc
import re
import traceback
from datetime import datetime, timezone
from config.config import config

# Import authentication services
from services.auth_service import AuthService
from services.session_manager import SessionManager  
from services.auth_decorators import require_auth, require_permission, optional_auth

# Import existing modules with error handling
try:
    from core.medgemma_inference import MedGemmaInference
    MEDGEMMA_AVAILABLE = True   
    logger.info("✅ MedGemma import successful")
except ImportError as e:
    MEDGEMMA_AVAILABLE = False
    logger.info(f"❌ MedGemma import failed: {e}")

try:
    from core.rag_system import RAGSystem
    RAG_AVAILABLE = True
    logger.info("✅ RAG import successful")
except ImportError as e:
    RAG_AVAILABLE = False
    logger.info(f"❌ RAG import failed: {e}")

logger.info(f"🔍 RAG_AVAILABLE flag: {RAG_AVAILABLE}")
logger.info(f"🔍 MEDGEMMA_AVAILABLE flag: {MEDGEMMA_AVAILABLE}")

def cleanup_temp_file(file_path, max_retries=5):
    """Clean up temporary file with Windows-compatible retry logic"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                if attempt > 0:
                    time.sleep(0.5)
                gc.collect()
                os.unlink(file_path)
                logger.info(f"✅ Temporary file cleaned up: {file_path}")
                return True
        except PermissionError:
            if attempt < max_retries - 1:
                logger.info(f"⚠️ File locked, retrying cleanup attempt {attempt + 1}/{max_retries}")
                time.sleep(1)
                continue
            else:
                logger.info(f"❌ Could not delete temporary file after {max_retries} attempts: {file_path}")
                return False
        except Exception as e:
            logger.info(f"❌ Unexpected error during cleanup: {e}")
            return False
    return False

def create_app(config_name='default'):
    """Application factory pattern with enhanced medical AI and robust error handling"""
    logger.info("🔍 DEBUG: Entering create_app() function")
    
    try:
        app = Flask(__name__)
        logger.info("🔍 DEBUG: Flask app instance created")
        
        app.config.from_object(config[config_name])
        logger.info("🔍 DEBUG: Config loaded successfully")
        
        # Initialize authentication services
        try:
            app.auth_service = AuthService()
            logger.info("✅ Authentication service initialized")
        except Exception as e:
            logger.info(f"❌ Authentication service failed: {e}")
            raise
        
        # Initialize MedGemma with GPU support
        if MEDGEMMA_AVAILABLE:
            try:
                logger.info("🔄 Initializing MedGemma with GPU support...")
                app.medgemma = MedGemmaInference("models/medgemma-4b-it-Q8_0.gguf")
                app.medgemma.start_background_loading()
                logger.info("✅ MedGemma initialization started")
                
                # Non-blocking status check
                try:
                    status = app.medgemma.get_loading_status()
                    logger.info(f"📊 Initial model status: {status}")
                    
                    if app.medgemma.is_ready():
                        logger.info("✅ Model is ready for generation")
                    else:
                        logger.info("⚠️ Model not ready yet - will continue loading in background")
                        
                except Exception as status_error:
                    logger.info(f"⚠️ Could not get model status: {status_error}")
                    
            except Exception as model_error:
                logger.info(f"❌ MedGemma initialization failed: {model_error}")
                traceback.logger.info_exc()
                app.medgemma = None
                
        else:
            app.medgemma = None
            logger.info("❌ MedGemma not available")
        
        # Initialize RAG system with ENHANCED error handling and debugging
        if RAG_AVAILABLE:
            try:
                logger.info("🔄 Creating enhanced RAG system instance...")
                
                # Ensure data directory exists with proper permissions
                data_dir = 'data/chromadb'
                os.makedirs(data_dir, exist_ok=True)
                logger.info(f"📁 Data directory created: {data_dir}")
                
                # Test directory permissions
                test_file = os.path.join(data_dir, 'test_write.tmp')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    logger.info("✅ Directory permissions OK")
                except Exception as perm_error:
                    logger.info(f"⚠️ Directory permission issue: {perm_error}")
                
                # Initialize RAG system with medical embeddings
                logger.info("🧠 Initializing with medical embeddings...")
                app.rag_system = RAGSystem(
                    data_dir=data_dir,
                    embedding_model="medical"  # Use medical embeddings
                )
                logger.info("✅ RAG system initialized successfully!")
                
                # Test basic functionality
                try:
                    stats = app.rag_system.get_system_stats()
                    logger.info(f"📊 RAG system stats: {stats}")
                except Exception as stats_error:
                    logger.info(f"⚠️ Could not get RAG stats: {stats_error}")
                
                # Enhanced prompt construction
                def enhanced_augment_prompt_with_context(self, original_prompt, context_docs):
                    """Improved prompt construction with better instructions and context handling"""
                    logger.info(f"🔍 Processing {len(context_docs)} context documents")
                    
                    # Only use documents with positive similarity (>0)
                    good_docs = [doc for doc in context_docs if doc.get('similarity', 0) > 0]
                    logger.info(f"🔍 Found {len(good_docs)} good documents (similarity > 0)")
                    
                    # Extract patient name for personalized prompts
                    patient_name = None
                    patient_patterns = [
                        r'Patient Name:\s*([^,\n]+(?:,\s*[^,\n]+)?)',
                        r'patient:\s*([^,\n]+(?:,\s*[^,\n]+)?)',
                        r'([A-Z][a-zA-Z]+,\s*[A-Z][a-zA-Z]+)'
                    ]
                    
                    for pattern in patient_patterns:
                        match = re.search(pattern, original_prompt, re.IGNORECASE)
                        if match:
                            patient_name = match.group(1).strip()
                            logger.info(f"🔍 Detected patient: {patient_name}")
                            break
                    
                    if not good_docs:
                        # Better fallback prompt when no relevant context
                        if patient_name:
                            fallback_prompt = f"""You are a medical AI assistant. Create a comprehensive medical report for {patient_name}.

Since no specific medical records are available, generate a structured medical report template with the following sections:

## MEDICAL REPORT: {patient_name}

### PATIENT IDENTIFICATION
- Patient Name: {patient_name}
- Age: [To be determined]
- Gender: [To be determined]
- Medical Record Number: [To be assigned]

### CHIEF COMPLAINT
[Patient's main reason for medical visit]

### MEDICAL HISTORY
[Past medical conditions, surgeries, chronic diseases]

### CURRENT SYMPTOMS
[Present symptoms and their characteristics]

### CURRENT MEDICATIONS
[List of current medications and dosages]

### CLINICAL FINDINGS
[Physical examination findings and vital signs]

### DIAGNOSTIC TESTS
[Laboratory results and imaging studies]

### ASSESSMENT AND PLAN
[Medical assessment and treatment recommendations]

Please provide detailed content for each section based on common medical presentation patterns."""
                        else:
                            fallback_prompt = f"""You are a medical AI assistant. Based on the request: "{original_prompt}"

Please provide a comprehensive medical response that includes:
- Detailed medical information
- Clinical considerations
- Treatment recommendations
- Follow-up care suggestions

Generate a thorough medical analysis addressing the request."""
                        
                        logger.info(f"✅ Using fallback prompt (length: {len(fallback_prompt)})")
                        return fallback_prompt
                    
                    # Build concise but informative context
                    context_text = ""
                    for i, doc in enumerate(good_docs[:3], 1):
                        doc_excerpt = doc['text'][:600] + "..." if len(doc['text']) > 600 else doc['text']
                        context_text += f"\n--- Medical Document {i} ---\n{doc_excerpt}\n"
                    
                    # More effective prompt structure
                    if patient_name:
                        enhanced_prompt = f"""You are a medical AI assistant. Using the medical records below, create a comprehensive medical report for {patient_name}.

AVAILABLE MEDICAL RECORDS:
{context_text}

TASK: Generate a detailed, structured medical report that includes:

## COMPREHENSIVE MEDICAL REPORT: {patient_name}

### PATIENT IDENTIFICATION
Extract and present patient demographics, identifiers, and basic information.

### CHIEF COMPLAINT & PRESENTING SYMPTOMS
Detail the main medical concerns, symptoms, onset, duration, and severity.

### MEDICAL HISTORY
Summarize past medical conditions, surgeries, hospitalizations, and chronic diseases.

### CURRENT DIAGNOSES
List primary and secondary diagnoses with supporting clinical evidence.

### MEDICATIONS & TREATMENTS
Document current medications (names, dosages, frequencies) and treatments.

### CLINICAL FINDINGS & TEST RESULTS
Report vital signs, laboratory values, imaging results, and examination findings.

### HEALTHCARE PROVIDER ASSESSMENTS
Include provider notes, clinical observations, and medical opinions.

### CARE PLAN & RECOMMENDATIONS
Outline treatment plans, follow-up care, monitoring, and specialist referrals.

INSTRUCTIONS:
- Write complete, detailed paragraphs for each section
- Use specific medical information from the records
- Include dates, values, and measurements when available
- Use professional medical terminology
- If information is missing, state "Not documented in available records"

Begin generating the comprehensive medical report now:"""
                    else:
                        enhanced_prompt = f"""You are a medical AI assistant. Based on the medical information provided below, answer the following question comprehensively:

MEDICAL INFORMATION:
{context_text}

QUESTION: {original_prompt}

INSTRUCTIONS:
- Provide a detailed, evidence-based medical response
- Use specific information from the medical records
- Include clinical reasoning and recommendations
- Use professional medical terminology
- Cite relevant findings from the documents

Generate a thorough medical response:"""
                    
                    # Safety: Manage prompt length
                    if len(enhanced_prompt) > 2000:
                        logger.info("⚠️ Prompt exceeds 2000 chars, using shorter version")
                        if patient_name:
                            return f"""Create a comprehensive medical report for {patient_name} using these medical records:

{context_text[:800]}

Include: patient identification, symptoms, medical history, diagnoses, medications, clinical findings, and care plan.

Generate detailed medical report:"""
                        else:
                            return f"Based on: {context_text[:500]}\n\nQuestion: {original_prompt}\n\nProvide detailed medical response:"
                    
                    logger.info(f"✅ Enhanced prompt created (length: {len(enhanced_prompt)})")
                    return enhanced_prompt

                # Bind the enhanced method
                import types
                app.rag_system.augment_prompt_with_context = types.MethodType(
                    enhanced_augment_prompt_with_context, 
                    app.rag_system
                )
                
                logger.info("✅ Enhanced RAG system with improved prompts initialized!")
                
                # Initialize session manager with RAG database
                app.session_manager = SessionManager(
                    db_manager=app.rag_system.db_manager if hasattr(app.rag_system, 'db_manager') else None,
                    session_timeout_hours=24
                )
                logger.info("✅ Session manager initialized with RAG integration")
                
            except ImportError as rag_import_error:
                logger.info(f"❌ RAG import error: {rag_import_error}")
                logger.info("📦 Missing dependencies. Install with:")
                logger.info("   pip install chromadb sentence-transformers langchain")
                app.rag_system = None
                app.session_manager = SessionManager()
                
            except Exception as rag_error:
                logger.info(f"❌ RAG system initialization failed: {rag_error}")
                logger.info("📍 Full error traceback:")
                traceback.logger.info_exc()
                
                # Try simple fallback
                logger.info("🔄 Attempting simple RAG fallback...")
                try:
                    class SimpleRAGFallback:
                        def __init__(self):
                            self.documents = []
                            logger.info("⚠️ Using simple RAG fallback")
                        
                        def ingest_document_from_file(self, file_path, doc_type='medical', language='en'):
                            return {'success': False, 'error': 'RAG system unavailable', 'chunks_created': 0}
                        
                        def search_relevant_context(self, query, max_docs=5):
                            return []
                        
                        def get_system_stats(self):
                            return {'total_documents': 0, 'status': 'Fallback mode'}
                    
                    app.rag_system = SimpleRAGFallback()
                    logger.info("✅ Simple RAG fallback initialized")
                    
                except Exception as fallback_error:
                    logger.info(f"❌ Even fallback failed: {fallback_error}")
                    app.rag_system = None
                
                app.session_manager = SessionManager()
        else:
            app.rag_system = None
            app.session_manager = SessionManager()
            logger.info("❌ RAG not available - module import failed")
        
        # Register routes
        register_routes(app)
        
        return app
        
    except Exception as e:
        logger.info(f"❌ CRITICAL ERROR in create_app(): {e}")
        traceback.logger.info_exc()
        raise

def register_routes(app):
    """Register all application routes"""
    
    @app.route('/')
    @optional_auth(app.session_manager)
    def home():
        """Main dashboard endpoint with detailed system status"""
        user_info = getattr(request, 'current_user', None)
        
        # Get detailed component status
        medgemma_status = "❌ Not Available"
        if MEDGEMMA_AVAILABLE and app.medgemma:
            try:
                status = app.medgemma.get_loading_status()
                if status["status"] == "loaded":
                    medgemma_status = "✅ Ready"
                elif status["status"] == "loading":
                    medgemma_status = f"🔄 Loading ({status['progress']}%)"
                elif status["status"] == "failed":
                    medgemma_status = "❌ Failed"
                else:
                    medgemma_status = f"{status['status']} ({status.get('progress', 0)}%)"
            except Exception as e:
                medgemma_status = f"❌ Status Error: {str(e)}"
        
        rag_status = "❌ Not Available"
        if RAG_AVAILABLE and app.rag_system:
            try:
                stats = app.rag_system.get_system_stats()
                rag_status = f"✅ {stats.get('status', 'Ready')}"
                if 'total_documents' in stats:
                    rag_status += f" ({stats['total_documents']} docs)"
            except Exception as e:
                rag_status = f"❌ Status Error: {str(e)}"
        elif RAG_AVAILABLE and not app.rag_system:
            rag_status = "❌ Import OK, Init Failed"
        
        return jsonify({
            "message": "🔬 PulseQuery AI - Complete Medical System",
            "status": "Running",
            "milestone": 4,
            "enhancement": "Complete system with debugging and medical embeddings",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "authenticated": user_info is not None,
            "user": user_info,
            "components": {
                "flask": "✅ Running",
                "medgemma": medgemma_status,
                "rag_system": rag_status,
                "auth_service": "✅ Ready" if app.auth_service else "❌ Failed",
                "session_manager": "✅ Ready" if app.session_manager else "❌ Failed",
                "enhanced_prompts": "✅ V2.0 Enabled",
                "document_upload": "✅ Available" if app.rag_system else "❌ RAG Required",
                "medical_embeddings": "✅ Enabled" if app.rag_system else "❌ Not Available"
            }
        })

    # Authentication endpoints
    @app.route('/api/auth/login', methods=['POST'])
    def login():
        """User login endpoint"""
        try:
            data = request.get_json()
            user_id = data.get('user_id')
            password = data.get('password')
            
            if not user_id or not password:
                return jsonify({
                    "success": False,
                    "error": "User ID and password required"
                }), 400
            
            user_info = app.auth_service.authenticate_user(user_id, password)
            if not user_info:
                return jsonify({
                    "success": False,
                    "error": "Invalid credentials"
                }), 401
            
            session_id = app.session_manager.create_session(user_info)
            session['session_id'] = session_id
            session['user_id'] = user_id
            
            return jsonify({
                "success": True,
                "message": "Login successful",
                "user": user_info,
                "session_id": session_id,
                "milestone": 4
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Login failed: {str(e)}"
            }), 500

    @app.route('/api/auth/logout', methods=['POST'])
    @require_auth(app.session_manager)
    def logout():
        """User logout endpoint"""
        try:
            session_id = session.get('session_id')
            if session_id:
                app.session_manager.terminate_session(session_id)
            session.clear()
            
            return jsonify({
                "success": True,
                "message": "Logout successful",
                "milestone": 4
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Logout failed: {str(e)}"
            }), 500

    @app.route('/health')
    @optional_auth(app.session_manager)
    def health_check():
        """Comprehensive system health check"""
        user_info = getattr(request, 'current_user', None)
        
        # Detailed component health
        health_status = {
            "status": "healthy",
            "authenticated": user_info is not None,
            "user": user_info,
            "components": {
                "flask": "✅ Running",
                "auth_service": "✅ Ready" if app.auth_service else "❌ Failed",
                "session_manager": "✅ Ready" if app.session_manager else "❌ Failed",
            },
            "milestone": 4,
            "debug_info": {
                "RAG_AVAILABLE": RAG_AVAILABLE,
                "MEDGEMMA_AVAILABLE": MEDGEMMA_AVAILABLE,
                "rag_system_instance": app.rag_system is not None,
                "medgemma_instance": app.medgemma is not None
            }
        }
        
        # MedGemma health
        if MEDGEMMA_AVAILABLE and app.medgemma:
            try:
                loading_status = app.medgemma.get_loading_status()
                if loading_status["status"] == "loaded":
                    health_status["components"]["medgemma"] = "✅ Ready"
                elif loading_status["status"] == "loading":
                    health_status["components"]["medgemma"] = f"🔄 Loading ({loading_status['progress']}%)"
                elif loading_status["status"] == "failed":
                    health_status["components"]["medgemma"] = "❌ Failed"
                else:
                    health_status["components"]["medgemma"] = f"⚠️ {loading_status['status']}"
            except Exception as e:
                health_status["components"]["medgemma"] = f"❌ Status Error: {str(e)}"
        else:
            health_status["components"]["medgemma"] = "❌ Not Available"
        
        # RAG system health
        if RAG_AVAILABLE and app.rag_system:
            try:
                stats = app.rag_system.get_system_stats()
                health_status["components"]["rag_system"] = f"✅ {stats.get('status', 'Ready')}"
                health_status["components"]["document_upload"] = "✅ Available"
                health_status["components"]["medical_embeddings"] = "✅ Active"
                health_status["debug_info"]["rag_stats"] = stats
            except Exception as e:
                health_status["components"]["rag_system"] = f"❌ Status Error: {str(e)}"
                health_status["components"]["document_upload"] = "❌ Unavailable"
        elif RAG_AVAILABLE and not app.rag_system:
            health_status["components"]["rag_system"] = "❌ Import OK, Init Failed"
            health_status["components"]["document_upload"] = "❌ RAG Failed"
        else:
            health_status["components"]["rag_system"] = "❌ Import Failed"
            health_status["components"]["document_upload"] = "❌ Not Available"
        
        return jsonify(health_status)

    # Model status and generation endpoints
    @app.route('/api/medgemma/status')
    @optional_auth(app.session_manager)
    def medgemma_status():
        """Get detailed MedGemma status"""
        if not MEDGEMMA_AVAILABLE or not app.medgemma:
            return jsonify({
                "status": "unavailable",
                "message": "MedGemma not available",
                "available": MEDGEMMA_AVAILABLE,
                "instance": app.medgemma is not None
            })
        
        try:
            status = app.medgemma.get_loading_status()
            return jsonify({
                "milestone": 4,
                **status,
                "gpu_support": "Enabled",
                "model_file": "medgemma-4b-it-Q8_0.gguf"
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e),
                "milestone": 4
            })

    @app.route('/api/medgemma/generate-simple', methods=['POST'])
    @require_auth(app.session_manager)
    def medgemma_generate_simple():
        """Simple generation with improved prompts"""
        if not MEDGEMMA_AVAILABLE or not app.medgemma:
            return jsonify({"error": "MedGemma not available"}), 503
        
        if not app.medgemma.is_ready():
            return jsonify({"error": "Model not ready"}), 503
        
        try:
            data = request.get_json()
            original_prompt = data.get('prompt', "Generate a medical summary.")
            
            # Enhanced simple prompt
            if "Patient Name:" in original_prompt:
                patient_match = re.search(r'Patient Name:\s*([^,\n]+)', original_prompt, re.IGNORECASE)
                if patient_match:
                    patient_name = patient_match.group(1).strip()
                    simple_prompt = f"""You are a medical AI assistant. Create a comprehensive medical report for {patient_name}.

Generate a detailed medical report with the following structure:

## MEDICAL REPORT: {patient_name}

### PATIENT IDENTIFICATION
- Patient Name: {patient_name}
- Age: [Based on available information]
- Gender: [To be determined from records]

### CHIEF COMPLAINT
[Main reason for medical visit or concern]

### MEDICAL HISTORY
[Past medical conditions, surgeries, chronic diseases]

### CURRENT SYMPTOMS
[Present symptoms and clinical presentation]

### MEDICATIONS
[Current medications and treatments]

### CLINICAL FINDINGS
[Physical examination and test results]

### ASSESSMENT AND PLAN
[Medical assessment and treatment recommendations]

Please provide detailed content for each section:"""
                else:
                    simple_prompt = f"You are a medical AI assistant. {original_prompt}\n\nProvide a comprehensive medical response:"
            else:
                simple_prompt = f"You are a medical AI assistant. Please provide a detailed response to: {original_prompt}"
            
            result = app.medgemma.generate_text(
                prompt=simple_prompt,
                max_tokens=500,
                temperature=0.4
            )
            
            return jsonify({
                "success": True,
                "result": result,
                "prompt_used": simple_prompt,
                "test_type": "improved_simple",
                "generated_by": request.current_user["user_name"]
            })
            
        except Exception as e:
            logger.info(f"❌ Simple generation failed: {e}")
            return jsonify({"error": str(e)}), 500

    # RAG system endpoints
    @app.route('/api/rag/stats')
    @require_auth(app.session_manager)
    def rag_stats():
        """Get comprehensive RAG statistics"""
        if not RAG_AVAILABLE or not app.rag_system:
            return jsonify({
                "error": "RAG system not available",
                "debug": {
                    "RAG_AVAILABLE": RAG_AVAILABLE,
                    "rag_system_instance": app.rag_system is not None
                }
            }), 503
        
        try:
            stats = app.rag_system.get_system_stats()
            return jsonify({
                "milestone": 4,
                "rag_stats": stats,
                "accessed_by": request.current_user["user_name"],
                "embedding_info": {
                    "model": stats.get('embedding_model', 'Unknown'),
                    "type": "Medical-specialized"
                }
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/rag/upload', methods=['POST'])
    @require_permission(app.session_manager, 'write')
    def rag_upload():
        """Enhanced document upload with comprehensive error handling"""
        logger.info(f"🔍 DEBUG: app.rag_system = {app.rag_system}")
        logger.info(f"🔍 DEBUG: RAG_AVAILABLE = {RAG_AVAILABLE}")
        if not RAG_AVAILABLE or not app.rag_system:
            return jsonify({
                "error": "RAG system not available",
                "debug": {
                    "RAG_AVAILABLE": RAG_AVAILABLE,
                    "rag_system_instance": app.rag_system is not None,
                    "suggestion": "Check server logs for RAG initialization errors"
                }
            }), 503
        
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            doc_type = request.form.get('doc_type', 'medical')
            language = request.form.get('language', 'en')
            
            # Create temporary file with unique ID
            unique_id = str(uuid.uuid4())[:8]
            file_extension = os.path.splitext(file.filename)[1]
            temp_filename = f"pulsequery_upload_{unique_id}{file_extension}"
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, temp_filename)
            
            logger.info(f"📤 Processing upload: {file.filename} -> {temp_path}")
            
            try:
                file.save(temp_path)
                time.sleep(0.1)  # Allow file system to sync
                
                # Process through RAG system with medical embeddings
                result = app.rag_system.ingest_document_from_file(
                    file_path=temp_path,
                    doc_type=doc_type,
                    language=language
                )
                
                # Add user and file info
                result.update({
                    "uploaded_by": request.current_user["user_name"],
                    "user_id": request.current_user["user_id"],
                    "original_filename": file.filename,
                    "doc_type": doc_type,
                    "language": language,
                    "processing_method": "Medical embeddings"
                })
                
                logger.info(f"✅ Upload result: {result}")
                
                return jsonify({
                    "success": result.get('success', True),
                    "message": "Document uploaded and processed successfully",
                    "milestone": 4,
                    **result
                })
                
            finally:
                # Always cleanup temp file
                cleanup_temp_file(temp_path)
        
        except Exception as e:
            logger.info(f"❌ Upload failed: {e}")
            traceback.logger.info_exc()
            return jsonify({
                "error": str(e),
                "debug": {
                    "file_name": file.filename if 'file' in locals() else 'unknown',
                    "temp_path": temp_path if 'temp_path' in locals() else 'unknown'
                }
            }), 500

    @app.route('/api/rag/search', methods=['POST'])
    @require_auth(app.session_manager)
    def rag_search():
        """Search documents with medical embeddings"""
        if not RAG_AVAILABLE or not app.rag_system:
            return jsonify({"error": "RAG system not available"}), 503
        
        try:
            data = request.get_json()
            query = data.get('query')
            max_docs = data.get('max_docs', 5)
            
            if not query:
                return jsonify({"error": "Query required"}), 400
            
            results = app.rag_system.search_relevant_context(query, max_docs=max_docs)
            
            return jsonify({
                "milestone": 4,
                "query": query,
                "results_count": len(results),
                "results": results,
                "searched_by": request.current_user["user_name"],
                "search_type": "Medical semantic search"
            })
        
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/medgemma/generate-with-rag', methods=['POST'])
    @require_auth(app.session_manager)
    def medgemma_generate_with_rag():
        """Generate with medical RAG context"""
        logger.info("🔍 Enhanced RAG Generate endpoint called")
        
        if not MEDGEMMA_AVAILABLE or not app.medgemma:
            return jsonify({"error": "MedGemma not available"}), 503
        
        if not RAG_AVAILABLE or not app.rag_system:
            return jsonify({"error": "RAG system not available"}), 503
        
        if not app.medgemma.is_ready():
            status = app.medgemma.get_loading_status()
            return jsonify({
                "error": "Model not ready", 
                "status": status["status"],
                "progress": status["progress"]
            }), 503
        
        try:
            data = request.get_json()
            prompt = data.get('prompt')
            max_tokens = min(data.get('max_tokens', 600), 800)
            temperature = data.get('temperature', 0.3)
            use_rag = data.get('use_rag', True)
            
            logger.info(f"🔍 Enhanced RAG params: prompt_len={len(prompt) if prompt else 0}, max_tokens={max_tokens}, temp={temperature}, use_rag={use_rag}")
            
            if not prompt:
                return jsonify({"error": "Prompt required"}), 400
            
            # Get context with medical embeddings
            context_docs = []
            final_prompt = prompt
            
            if use_rag:
                context_docs = app.rag_system.search_relevant_context(prompt, max_docs=5)
                logger.info(f"   Found {len(context_docs)} context documents")
                
                # Log similarity scores
                for i, doc in enumerate(context_docs):
                    sim_score = doc.get('similarity', 0)
                    logger.info(f"   Doc {i+1}: {sim_score:.3f} similarity (medical)")
                
                # Use medical-enhanced prompt construction
                final_prompt = app.rag_system.augment_prompt_with_context(prompt, context_docs)
                logger.info(f"   Final prompt length: {len(final_prompt)} characters")
                
                # Safety check
                estimated_tokens = len(final_prompt) // 4
                if estimated_tokens + max_tokens > 1900:
                    max_tokens = max(200, 1900 - estimated_tokens)
                    logger.info(f"   Adjusted max_tokens to: {max_tokens}")
            
            # Generate with medical context
            result = app.medgemma.generate_text(
                prompt=final_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = result.get('generated_text', '')
            logger.info(f"✅ Enhanced generation complete: {len(generated_text)} characters")
            
            if not generated_text:
                logger.info("❌ WARNING: Generated text is empty!")
                # Fallback with simpler prompt
                fallback_prompt = f"Create a medical report for: {prompt}"
                fallback_result = app.medgemma.generate_text(
                    prompt=fallback_prompt,
                    max_tokens=300,
                    temperature=0.5
                )
                
                if fallback_result.get('generated_text'):
                    result = fallback_result
                    generated_text = result.get('generated_text', '')
                    logger.info(f"✅ Fallback successful: {len(generated_text)} characters")
            
            return jsonify({
                "milestone": 4,
                "success": True,
                "original_prompt": prompt,
                "enhanced_prompt_used": use_rag,
                "final_prompt_length": len(final_prompt),
                "context_docs_used": len(context_docs),
                "result": result,
                "generated_by": request.current_user["user_name"],
                "medical_context": True,
                "debug_info": {
                    "estimated_input_tokens": len(final_prompt) // 4,
                    "max_tokens_requested": max_tokens,
                    "similarity_scores": [doc.get('similarity', 0) for doc in context_docs],
                    "good_docs_count": len([d for d in context_docs if d.get('similarity', 0) > 0]),
                    "prompt_version": "v2.0_medical"
                }
            })
            
        except Exception as e:
            logger.info(f"❌ Enhanced RAG generation failed: {e}")
            traceback.logger.info_exc()
            return jsonify({"error": str(e)}), 500

    # Complete Test UI with all functionality
    @app.route('/test-ui')
    def test_ui():
        """Complete test interface with medical AI features"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PulseQuery AI - Complete Medical System</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; }
                .container { max-width: 1400px; margin: 0 auto; }
                .card { border: 1px solid #dee2e6; padding: 20px; margin: 15px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); background: white; }
                .card-header { background: linear-gradient(135deg, #007bff, #0056b3); color: white; padding: 15px; margin: -20px -20px 20px -20px; border-radius: 8px 8px 0 0; }
                .btn { padding: 10px 20px; border: none; cursor: pointer; margin: 5px; border-radius: 5px; font-weight: 500; transition: all 0.2s; }
                .btn-success { background: #28a745; color: white; } .btn-success:hover { background: #218838; }
                .btn-warning { background: #ffc107; color: #856404; } .btn-warning:hover { background: #e0a800; }
                .btn-danger { background: #dc3545; color: white; } .btn-danger:hover { background: #c82333; }
                .btn-info { background: #17a2b8; color: white; } .btn-info:hover { background: #138496; }
                .btn-primary { background: #007bff; color: white; } .btn-primary:hover { background: #0056b3; }
                .alert { padding: 15px; margin: 15px 0; border-radius: 5px; border: 1px solid transparent; }
                .alert-success { background: #d4edda; border-color: #c3e6cb; color: #155724; }
                .alert-info { background: #d1ecf1; border-color: #bee5eb; color: #0c5460; }
                .alert-warning { background: #fff3cd; border-color: #ffeaa7; color: #856404; }
                .alert-danger { background: #f8d7da; border-color: #f5c6cb; color: #721c24; }
                .form-control, .form-select { border-radius: 5px; border: 1px solid #ced4da; padding: 8px 12px; }
                pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; border: 1px solid #e9ecef; }
                .status-badge { padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
                .status-ready { background: #d4edda; color: #155724; }
                .status-loading { background: #fff3cd; color: #856404; }
                .status-error { background: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Header -->
                <div class="card">
                    <div class="card-header">
                        <h1>🏥 PulseQuery AI - Complete Medical System</h1>
                        <p class="mb-0">Advanced Medical AI with GPU Support, Document Processing & Medical Embeddings</p>
                    </div>
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>Version:</strong> 2.0 with MedEmbed Integration</p>
                            <p><strong>Status:</strong> <span id="systemStatus" class="status-badge">Checking...</span></p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>Features:</strong> GPU Acceleration, Medical Embeddings, RAG System</p>
                            <p><strong>Model:</strong> MedGemma 4B + Medical Document Processing</p>
                        </div>
                    </div>
                </div>

                <!-- System Health Dashboard -->
                <div class="card">
                    <div class="card-header">
                        <h5>🔧 System Health Dashboard</h5>
                    </div>
                    <div class="row">
                        <div class="col-md-3">
                            <button class="btn btn-info w-100" onclick="testConnection()">🔗 Connection Test</button>
                        </div>
                        <div class="col-md-3">
                            <button class="btn btn-info w-100" onclick="checkHealth()">❤️ Full Health Check</button>
                        </div>
                        <div class="col-md-3">
                            <button class="btn btn-info w-100" onclick="checkModel()">🤖 Model Status</button>
                        </div>
                        <div class="col-md-3">
                            <button class="btn btn-info w-100" onclick="checkRAG()">🔍 RAG System</button>
                        </div>
                    </div>
                    <div id="systemResults" class="mt-3"></div>
                </div>

                <!-- Authentication -->
                <div class="card">
                    <div class="card-header">
                        <h5>🔐 User Authentication</h5>
                    </div>
                    <div class="row align-items-end">
                        <div class="col-md-3">
                            <label class="form-label">User Account:</label>
                            <select id="userId" class="form-select">
                                <option value="">Select User</option>
                                <option value="doctor1">👨‍⚕️ Dr. Alice Johnson (Full Access)</option>
                                <option value="admin1">👩‍💼 Admin Eve Wilson (Admin)</option>
                                <option value="nurse1">👩‍⚕️ Nurse Bob Smith (Read Only)</option>
                                <option value="resident1">👨‍⚕️ Dr. Charlie Brown (Resident)</option>
                            </select>
                        </div>
                        <div class="col-md-3">
                            <label class="form-label">Password:</label>
                            <input type="password" id="password" class="form-control" placeholder="Password">
                        </div>
                        <div class="col-md-3">
                            <button class="btn btn-success" onclick="login()">🔐 Login</button>
                            <button class="btn btn-warning" onclick="logout()">🚪 Logout</button>
                        </div>
                        <div class="col-md-3">
                            <div class="alert alert-info mb-0 py-2">
                                <small id="loginStatus">Not logged in</small>
                            </div>
                        </div>
                    </div>
                    <small class="text-muted mt-2">Demo Passwords: password123, admin123, nurse123, resident123</small>
                </div>

                <!-- Document Upload (Medical Records) -->
                <div class="card" id="uploadCard" style="display: none;">
                    <div class="card-header">
                        <h5>📄 Medical Document Upload</h5>
                        <small>Upload patient records, clinical protocols, or research papers for AI analysis</small>
                    </div>
                    <form id="uploadForm">
                        <div class="row">
                            <div class="col-md-4">
                                <label class="form-label">Select Document:</label>
                                <input type="file" class="form-control" id="documentFile" accept=".pdf,.docx,.txt" required>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Document Type:</label>
                                <select class="form-select" id="docType">
                                    <option value="medical">📋 Medical Record</option>
                                    <option value="protocol">🔬 Clinical Protocol</option>
                                    <option value="guideline">📚 Medical Guideline</option>
                                    <option value="research">🧪 Research Paper</option>
                                    <option value="report">📊 Lab Report</option>
                                </select>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Language:</label>
                                <select class="form-select" id="docLanguage">
                                    <option value="en">🇺🇸 English</option>
                                    <option value="es">🇪🇸 Spanish</option>
                                    <option value="fr">🇫🇷 French</option>
                                </select>
                            </div>
                            <div class="col-md-2">
                                <label class="form-label">&nbsp;</label>
                                <button type="submit" class="btn btn-primary w-100">📤 Upload & Process</button>
                            </div>
                        </div>
                    </form>
                    <div id="uploadResult" class="mt-3" style="display: none;"></div>
                </div>

                <!-- Document Search -->
                <div class="card" id="searchCard" style="display: none;">
                    <div class="card-header">
                        <h5>🔍 Medical Document Search</h5>
                        <small>Search uploaded documents using medical-specialized AI embeddings</small>
                    </div>
                    <div class="row">
                        <div class="col-md-8">
                            <input type="text" class="form-control" id="searchQuery" placeholder="Enter medical search query (e.g., 'diabetes management', 'chest pain symptoms')..." value="diabetes symptoms and treatment">
                        </div>
                        <div class="col-md-2">
                            <input type="number" class="form-control" id="maxResults" value="5" min="1" max="10" placeholder="Max results">
                        </div>
                        <div class="col-md-2">
                            <button class="btn btn-info w-100" onclick="searchDocuments()">🔍 Search</button>
                        </div>
                    </div>
                    <div id="searchResults" class="mt-3" style="display: none;"></div>
                </div>

                <!-- AI Generation Testing -->
                <div class="card" id="generationCard" style="display: none;">
                    <div class="card-header">
                        <h5>🧠 Medical AI Generation</h5>
                        <small>Test medical report generation with and without document context</small>
                    </div>
                    <div class="row mb-3">
                        <div class="col-md-8">
                            <label class="form-label">Medical Prompt:</label>
                            <textarea id="promptText" class="form-control" rows="3" placeholder="Enter medical prompt...">Patient Name: Rogers, Pamela
Chief Complaint: Chest pain and shortness of breath</textarea>
                        </div>
                        <div class="col-md-4">
                            <label class="form-label">Generation Settings:</label>
                            <input type="number" id="maxTokens" class="form-control mb-2" value="600" min="100" max="800" placeholder="Max Tokens">
                            <input type="number" id="temperature" class="form-control" value="0.3" min="0.1" max="1.0" step="0.1" placeholder="Temperature">
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-12">
                            <button class="btn btn-info" onclick="testSimpleGeneration()">🧪 Simple Generation</button>
                            <button class="btn btn-success" onclick="testRAGGeneration()">🔬 RAG-Enhanced Generation</button>
                            <button class="btn btn-warning" onclick="searchAndGenerate()">🔍➕🧠 Search + Generate</button>
                            <button class="btn btn-danger" onclick="clearResults()">🗑️ Clear Results</button>
                        </div>
                    </div>
                    <div id="generationResults" class="mt-3"></div>
                </div>

                <!-- Footer -->
                <div class="card">
                    <div class="text-center">
                        <p class="mb-2"><strong>PulseQuery AI</strong> - Advanced Medical AI System</p>
                        <p class="mb-0 text-muted">Powered by MedGemma 4B, Medical Embeddings, and RAG Technology</p>
                    </div>
                </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                let currentUser = null;
                let sessionId = null;

                document.addEventListener('DOMContentLoaded', function() {
                    console.log('🏥 PulseQuery AI Complete Medical System Loading...');
                    updateSystemStatus();
                    checkHealth(); // Initial health check
                });

                function updateSystemStatus() {
                    fetch('/')
                    .then(response => response.json())
                    .then(data => {
                        const statusElement = document.getElementById('systemStatus');
                        statusElement.textContent = 'Running ✅';
                        statusElement.className = 'status-badge status-ready';
                        console.log('System status:', data);
                    })
                    .catch(error => {
                        const statusElement = document.getElementById('systemStatus');
                        statusElement.textContent = 'Error ❌';
                        statusElement.className = 'status-badge status-error';
                    });
                }

                function testConnection() {
                    document.getElementById('systemResults').innerHTML = '<div class="alert alert-info">🔄 Testing connection...</div>';
                    fetch('/')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('systemResults').innerHTML = 
                            '<div class="alert alert-success">' +
                            '<h6>✅ Connection Test Successful</h6>' +
                            '<p><strong>Message:</strong> ' + data.message + '</p>' +
                            '<p><strong>Timestamp:</strong> ' + new Date(data.timestamp).toLocaleString() + '</p>' +
                            '</div>';
                    })
                    .catch(error => {
                        document.getElementById('systemResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ Connection Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function checkHealth() {
                    document.getElementById('systemResults').innerHTML = '<div class="alert alert-info">🔄 Performing comprehensive health check...</div>';
                    fetch('/health')
                    .then(response => response.json())
                    .then(data => {
                        let html = '<div class="alert alert-success"><h6>❤️ System Health Check</h6>';
                        
                        if (data.components) {
                            html += '<div class="row mt-3">';
                            Object.entries(data.components).forEach(([key, value]) => {
                                const status = value.includes('✅') ? 'success' : value.includes('🔄') ? 'warning' : 'danger';
                                html += '<div class="col-md-6 mb-2">';
                                html += '<span class="badge bg-' + status + ' me-2">' + key.replace('_', ' ') + '</span>';
                                html += '<span>' + value + '</span>';
                                html += '</div>';
                            });
                            html += '</div>';
                        }
                        
                        if (data.debug_info) {
                            html += '<details class="mt-3"><summary>🔧 Debug Information</summary>';
                            html += '<pre>' + JSON.stringify(data.debug_info, null, 2) + '</pre>';
                            html += '</details>';
                        }
                        
                        html += '</div>';
                        document.getElementById('systemResults').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('systemResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ Health Check Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function checkModel() {
                    document.getElementById('systemResults').innerHTML = '<div class="alert alert-info">🔄 Checking model status...</div>';
                    fetch('/api/medgemma/status')
                    .then(response => response.json())
                    .then(data => {
                        let html = '<div class="alert alert-info"><h6>🤖 MedGemma Model Status</h6>';
                        html += '<p><strong>Status:</strong> ' + data.status + '</p>';
                        html += '<p><strong>Progress:</strong> ' + (data.progress || 0) + '%</p>';
                        html += '<p><strong>Device:</strong> ' + (data.device || 'Unknown') + '</p>';
                        html += '<p><strong>GPU Support:</strong> ' + (data.gpu_support || 'Unknown') + '</p>';
                        html += '<p><strong>Model File:</strong> ' + (data.model_file || 'Unknown') + '</p>';
                        html += '</div>';
                        document.getElementById('systemResults').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('systemResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ Model Check Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function checkRAG() {
                    document.getElementById('systemResults').innerHTML = '<div class="alert alert-info">🔄 Checking RAG system...</div>';
                    fetch('/api/rag/stats')
                    .then(response => response.json())
                    .then(data => {
                        let html = '<div class="alert alert-success"><h6>🔍 RAG System Status</h6>';
                        if (data.rag_stats) {
                            Object.entries(data.rag_stats).forEach(([key, value]) => {
                                html += '<p><strong>' + key.replace('_', ' ') + ':</strong> ' + value + '</p>';
                            });
                        }
                        if (data.embedding_info) {
                            html += '<h6 class="mt-3">🧠 Embedding Information</h6>';
                            Object.entries(data.embedding_info).forEach(([key, value]) => {
                                html += '<p><strong>' + key + ':</strong> ' + value + '</p>';
                            });
                        }
                        html += '</div>';
                        document.getElementById('systemResults').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('systemResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ RAG Check Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function login() {
                    const userId = document.getElementById('userId').value;
                    const password = document.getElementById('password').value;
                    
                    if (!userId || !password) {
                        alert('Please select user and enter password');
                        return;
                    }
                    
                    fetch('/api/auth/login', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({user_id: userId, password: password})
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            currentUser = data.user;
                            sessionId = data.session_id;
                            document.getElementById('loginStatus').innerHTML = 
                                '<strong>Logged in as:</strong><br>' + data.user.name + '<br><small>(' + data.user.role + ')</small>';
                            
                            // Show appropriate components based on permissions
                            document.getElementById('searchCard').style.display = 'block';
                            document.getElementById('generationCard').style.display = 'block';
                            
                            if (data.user.permissions && data.user.permissions.includes('write')) {
                                document.getElementById('uploadCard').style.display = 'block';
                            }
                            
                            alert('✅ Login successful! Welcome, ' + data.user.name);
                        } else {
                            alert('❌ Login failed: ' + data.error);
                        }
                    })
                    .catch(error => alert('❌ Login error: ' + error));
                }

                function logout() {
                    if (!sessionId) return;
                    fetch('/api/auth/logout', {
                        method: 'POST',
                        headers: {'X-Session-ID': sessionId}
                    })
                    .then(() => {
                        currentUser = null;
                        sessionId = null;
                        document.getElementById('loginStatus').innerHTML = 'Not logged in';
                        document.getElementById('uploadCard').style.display = 'none';
                        document.getElementById('searchCard').style.display = 'none';
                        document.getElementById('generationCard').style.display = 'none';
                        alert('✅ Logged out successfully');
                    });
                }

                // Document upload functionality
                document.getElementById('uploadForm').addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    if (!sessionId) {
                        alert('Please login first');
                        return;
                    }
                    
                    const fileInput = document.getElementById('documentFile');
                    const docType = document.getElementById('docType').value;
                    const docLanguage = document.getElementById('docLanguage').value;
                    
                    if (!fileInput.files[0]) {
                        alert('Please select a file');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    formData.append('doc_type', docType);
                    formData.append('language', docLanguage);
                    
                    document.getElementById('uploadResult').innerHTML = 
                        '<div class="alert alert-info">' +
                        '<h6>📤 Processing Medical Document...</h6>' +
                        '<p>Uploading and analyzing with medical AI embeddings...</p>' +
                        '</div>';
                    document.getElementById('uploadResult').style.display = 'block';
                    
                    fetch('/api/rag/upload', {
                        method: 'POST',
                        headers: {
                            'X-Session-ID': sessionId
                        },
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('uploadResult').innerHTML = 
                                '<div class="alert alert-success">' +
                                '<h6>✅ Medical Document Processed Successfully!</h6>' +
                                '<div class="row">' +
                                '<div class="col-md-6">' +
                                '<p><strong>📄 File:</strong> ' + (data.original_filename || 'Unknown') + '</p>' +
                                '<p><strong>📊 Chunks Created:</strong> ' + (data.chunks_created || 0) + '</p>' +
                                '<p><strong>🔬 Document Type:</strong> ' + (data.doc_type || 'medical') + '</p>' +
                                '</div>' +
                                '<div class="col-md-6">' +
                                '<p><strong>👤 Uploaded by:</strong> ' + (data.uploaded_by || currentUser.name) + '</p>' +
                                '<p><strong>🧠 Processing Method:</strong> ' + (data.processing_method || 'Standard') + '</p>' +
                                '<p><strong>🌐 Language:</strong> ' + (data.language || 'en') + '</p>' +
                                '</div>' +
                                '</div>' +
                                '</div>';
                            
                            // Reset form
                            fileInput.value = '';
                        } else {
                            document.getElementById('uploadResult').innerHTML = 
                                '<div class="alert alert-danger">' +
                                '<h6>❌ Upload Failed</h6>' +
                                '<p><strong>Error:</strong> ' + (data.error || 'Unknown error') + '</p>' +
                                (data.debug ? '<details><summary>Debug Info</summary><pre>' + JSON.stringify(data.debug, null, 2) + '</pre></details>' : '') +
                                '</div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('uploadResult').innerHTML = 
                            '<div class="alert alert-danger">' +
                            '<h6>❌ Upload Error</h6>' +
                            '<p>' + error + '</p>' +
                            '</div>';
                    });
                });

                function searchDocuments() {
                    if (!sessionId) { alert('Please login first'); return; }
                    
                    const query = document.getElementById('searchQuery').value;
                    const maxResults = parseInt(document.getElementById('maxResults').value);
                    
                    if (!query.trim()) {
                        alert('Please enter a search query');
                        return;
                    }
                    
                    document.getElementById('searchResults').innerHTML = 
                        '<div class="alert alert-info"><h6>🔍 Searching medical documents...</h6><p>Using medical-specialized embeddings...</p></div>';
                    document.getElementById('searchResults').style.display = 'block';
                    
                    fetch('/api/rag/search', {
                        method: 'POST',
                        headers: { 
                            'Content-Type': 'application/json',
                            'X-Session-ID': sessionId 
                        },
                        body: JSON.stringify({ query: query, max_docs: maxResults })
                    })
                    .then(response => response.json())
                    .then(data => {
                        const resultsDiv = document.getElementById('searchResults');
                        if (data.results_count > 0) {
                            let html = '<div class="alert alert-success">';
                            html += '<h6>✅ Found ' + data.results_count + ' relevant medical documents</h6>';
                            html += '<p><strong>Query:</strong> ' + data.query + '</p>';
                            html += '<p><strong>Search Type:</strong> ' + (data.search_type || 'Semantic') + '</p>';
                            html += '</div>';
                            
                            data.results.forEach((result, index) => {
                                const relevancePercent = Math.round(result.similarity * 100);
                                const relevanceClass = relevancePercent > 70 ? 'success' : relevancePercent > 40 ? 'warning' : 'secondary';
                                
                                html += '<div class="card mb-3">';
                                html += '<div class="card-header d-flex justify-content-between align-items-center">';
                                html += '<h6 class="mb-0">📄 Document ' + (index + 1) + '</h6>';
                                html += '<span class="badge bg-' + relevanceClass + '">' + relevancePercent + '% Relevant</span>';
                                html += '</div>';
                                html += '<div class="card-body">';
                                html += '<p class="card-text">' + result.text.substring(0, 300) + '...</p>';
                                html += '<div class="row">';
                                html += '<div class="col-md-6">';
                                html += '<small class="text-muted"><strong>File:</strong> ' + (result.metadata.file_name || 'Unknown') + '</small><br>';
                                html += '<small class="text-muted"><strong>Type:</strong> ' + (result.metadata.doc_type || 'medical') + '</small>';
                                html += '</div>';
                                html += '<div class="col-md-6">';
                                html += '<small class="text-muted"><strong>Chunk:</strong> ' + (result.metadata.chunk_index || 0) + '/' + (result.metadata.total_chunks || 1) + '</small><br>';
                                html += '<small class="text-muted"><strong>Similarity:</strong> ' + result.similarity.toFixed(3) + '</small>';
                                html += '</div>';
                                html += '</div>';
                                html += '</div>';
                                html += '</div>';
                            });
                            
                            resultsDiv.innerHTML = html;
                        } else {
                            resultsDiv.innerHTML = '<div class="alert alert-warning"><h6>⚠️ No relevant documents found</h6><p>Try different search terms or upload more documents.</p></div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('searchResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ Search Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function testSimpleGeneration() {
                    if (!sessionId) { alert('Please login first'); return; }
                    
                    const prompt = document.getElementById('promptText').value;
                    document.getElementById('generationResults').innerHTML = 
                        '<div class="alert alert-info"><h6>🧪 Testing simple medical generation...</h6><p>Generating without document context...</p></div>';
                    
                    fetch('/api/medgemma/generate-simple', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json', 'X-Session-ID': sessionId},
                        body: JSON.stringify({prompt: prompt})
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success && data.result.generated_text) {
                            document.getElementById('generationResults').innerHTML = 
                                '<div class="alert alert-success">' +
                                '<h6>✅ Simple Generation Successful</h6>' +
                                '<div class="row mb-3">' +
                                '<div class="col-md-6"><strong>Length:</strong> ' + data.result.generated_text.length + ' characters</div>' +
                                '<div class="col-md-6"><strong>Generated by:</strong> ' + (data.generated_by || 'AI') + '</div>' +
                                '</div>' +
                                '<div class="card">' +
                                '<div class="card-header"><strong>Generated Medical Report:</strong></div>' +
                                '<div class="card-body"><pre>' + data.result.generated_text + '</pre></div>' +
                                '</div>' +
                                '</div>';
                        } else {
                            document.getElementById('generationResults').innerHTML = 
                                '<div class="alert alert-warning"><h6>⚠️ Empty Generation Response</h6><p>The model returned an empty response. Try adjusting the prompt.</p></div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('generationResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ Simple Generation Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function testRAGGeneration() {
                    if (!sessionId) { alert('Please login first'); return; }
                    
                    const prompt = document.getElementById('promptText').value;
                    const maxTokens = parseInt(document.getElementById('maxTokens').value);
                    const temperature = parseFloat(document.getElementById('temperature').value);
                    
                    document.getElementById('generationResults').innerHTML = 
                        '<div class="alert alert-info"><h6>🔬 Testing RAG-enhanced medical generation...</h6><p>Searching documents and generating with medical context...</p></div>';
                    
                    fetch('/api/medgemma/generate-with-rag', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json', 'X-Session-ID': sessionId},
                        body: JSON.stringify({
                            prompt: prompt,
                            max_tokens: maxTokens,
                            temperature: temperature,
                            use_rag: true
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success && data.result.generated_text) {
                            document.getElementById('generationResults').innerHTML = 
                                '<div class="alert alert-success">' +
                                '<h6>✅ RAG-Enhanced Generation Successful</h6>' +
                                '<div class="row mb-3">' +
                                '<div class="col-md-4"><strong>Length:</strong> ' + data.result.generated_text.length + ' chars</div>' +
                                '<div class="col-md-4"><strong>Context Docs:</strong> ' + data.context_docs_used + '</div>' +
                                '<div class="col-md-4"><strong>Good Docs:</strong> ' + (data.debug_info.good_docs_count || 0) + '</div>' +
                                '</div>' +
                                '<div class="card mb-3">' +
                                '<div class="card-header"><strong>Medical AI Generated Report (with RAG context):</strong></div>' +
                                '<div class="card-body"><pre>' + data.result.generated_text + '</pre></div>' +
                                '</div>' +
                                '<details>' +
                                '<summary>🔧 Generation Details</summary>' +
                                '<div class="mt-2">' +
                                '<p><strong>Input Tokens:</strong> ' + data.debug_info.estimated_input_tokens + '</p>' +
                                '<p><strong>Similarity Scores:</strong> ' + JSON.stringify(data.debug_info.similarity_scores) + '</p>' +
                                '<p><strong>Medical Context:</strong> ' + (data.medical_context ? 'Yes' : 'No') + '</p>' +
                                '</div>' +
                                '</details>' +
                                '</div>';
                        } else {
                            let debugInfo = '';
                            if (data.debug_info) {
                                debugInfo = '<div class="mt-3">' +
                                           '<p><strong>Debug Info:</strong></p>' +
                                           '<p>Input tokens: ' + data.debug_info.estimated_input_tokens + '</p>' +
                                           '<p>Good docs: ' + data.debug_info.good_docs_count + '</p>' +
                                           '<p>Similarities: ' + JSON.stringify(data.debug_info.similarity_scores) + '</p>' +
                                           '</div>';
                            }
                            document.getElementById('generationResults').innerHTML = 
                                '<div class="alert alert-warning"><h6>⚠️ Empty RAG Response</h6><p>No content was generated. This might be due to insufficient context or model issues.</p>' + debugInfo + '</div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('generationResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ RAG Generation Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function searchAndGenerate() {
                    // First search, then generate with top results
                    const query = document.getElementById('promptText').value;
                    
                    document.getElementById('generationResults').innerHTML = 
                        '<div class="alert alert-info"><h6>🔍➕🧠 Search and Generate</h6><p>Step 1: Searching for relevant documents...</p></div>';
                    
                    // Perform search first
                    fetch('/api/rag/search', {
                        method: 'POST',
                        headers: { 
                            'Content-Type': 'application/json',
                            'X-Session-ID': sessionId 
                        },
                        body: JSON.stringify({ query: query, max_docs: 3 })
                    })
                    .then(response => response.json())
                    .then(searchData => {
                        document.getElementById('generationResults').innerHTML = 
                            '<div class="alert alert-info"><h6>🔍➕🧠 Search and Generate</h6><p>Step 2: Found ' + searchData.results_count + ' documents. Generating comprehensive report...</p></div>';
                        
                        // Then generate with RAG
                        return testRAGGeneration();
                    })
                    .catch(error => {
                        document.getElementById('generationResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ Search and Generate Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function clearResults() {
                    document.getElementById('generationResults').innerHTML = '';
                    document.getElementById('systemResults').innerHTML = '';
                    document.getElementById('uploadResult').innerHTML = '';
                    document.getElementById('searchResults').innerHTML = '';
                    document.getElementById('uploadResult').style.display = 'none';
                    document.getElementById('searchResults').style.display = 'none';
                }
            </script>
        </body>
        </html>
        """

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found", "milestone": 4}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "milestone": 4}), 500

# Main execution
if __name__ == '__main__':
    logger.info("🚀 Starting PulseQuery AI - Complete Medical System")
    logger.info("🔧 Features: GPU Support, Medical Embeddings, Document Upload")
    logger.info("🧠 Enhanced: RAG System with Comprehensive Error Handling")
    logger.info("📄 NEW: Complete Document Processing Pipeline")
    logger.info("🔐 AUTH: Role-based Permissions and Session Management")
    logger.info("🎯 UI: Complete Testing Interface with Medical Features")
    
    try:
        logger.info("\n🔄 Creating complete Flask application...")
        app = create_app()
        logger.info("✅ Complete application created successfully!")
        
        logger.info("\n🌐 Server Information:")
        logger.info("📍 Main Server: http://localhost:5000")
        logger.info("🔗 Complete Test UI: http://localhost:5000/test-ui")
        logger.info("❤️ Health Check: http://localhost:5000/health")
        
        logger.info("\n🧪 Available Endpoints:")
        logger.info("   - /api/auth/login - User authentication")
        logger.info("   - /api/rag/upload - Document upload (write permission)")
        logger.info("   - /api/rag/search - Document search with medical embeddings")
        logger.info("   - /api/rag/stats - RAG system statistics")
        logger.info("   - /api/medgemma/status - Model status")
        logger.info("   - /api/medgemma/generate-simple - Simple generation")
        logger.info("   - /api/medgemma/generate-with-rag - RAG-enhanced generation")
        
        logger.info("\n👥 Demo Login Credentials:")
        logger.info("   - doctor1 / password123 (✅ Full access + document upload)")
        logger.info("   - admin1 / admin123 (✅ Administrative access)")  
        logger.info("   - nurse1 / nurse123 (📖 Read-only access)")
        logger.info("   - resident1 / resident123 (📖 Limited access)")
        
        logger.info("\n🔧 System Components:")
        logger.info("   - MedGemma 4B with GPU support")
        logger.info("   - Medical embeddings (MedEmbed/BioBERT)")
        logger.info("   - RAG system with ChromaDB")
        logger.info("   - Document processing (PDF, DOCX, TXT)")
        logger.info("   - Enhanced prompt engineering")
        logger.info("   - Complete authentication system")
        
        logger.info("\n🔄 Starting server with enhanced debugging...")
        
        app.run(
            debug=True,
            host=app.config.get('HOST', '127.0.0.1'), 
            port=app.config.get('PORT', 5000),
            use_reloader=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Server stopped by user")
    except Exception as e:
        logger.info(f"\n❌ Server startup failed: {e}")
        traceback.logger.info_exc()
    finally:
        logger.info("\n✅ Shutdown complete")
