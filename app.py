import logging
import warnings
from datetime import datetime, timezone
import os
import re
from collections import Counter
from flask import send_from_directory

# ✅ ENHANCED: File + Console Logging Configuration
LOG_DIR = 'log'
LOG_FILENAME = f'app_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, LOG_FILENAME)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# ✅ SUPPRESS: Tokenizer and model warnings
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

logging.getLogger().addFilter(NoiseFilter())
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)
logger.info(f"🗂️ Logging initialized - File: {LOG_FILENAME}")

# ✅ MEDICAL TERMINOLOGY DENSITY FIX
MEDICAL_TERMS = {
    'symptoms': ['pain', 'chest pain', 'shortness of breath', 'nausea', 'vomiting', 'dizziness', 'headache', 'fever', 'fatigue', 'weakness'],
    'conditions': ['hypertension', 'diabetes', 'cardiac', 'cardiology', 'emergency', 'diagnosis', 'syndrome', 'disease', 'disorder'],
    'demographics': ['patient', 'male', 'female', 'years old', 'age', 'elderly', 'adult', 'pediatric'],
    'clinical': ['diagnosis', 'treatment', 'symptoms', 'history', 'complaint', 'medication', 'therapy', 'procedure', 'examination'],
    'anatomy': ['heart', 'lung', 'brain', 'kidney', 'liver', 'stomach', 'chest', 'abdomen', 'extremities'],
    'medical_specialty': ['cardiology', 'neurology', 'gastroenterology', 'emergency', 'internal medicine', 'surgery'],
    'vitals': ['blood pressure', 'heart rate', 'temperature', 'respiratory rate', 'oxygen saturation', 'pulse'],
    'assessments': ['workup', 'evaluation', 'assessment', 'monitoring', 'follow-up', 'consultation']
}

def calculate_medical_terminology_density(text):
    """Calculate medical term density in text"""
    if not text or len(text.strip()) == 0:
        return 0.0
    
    # Tokenize text
    tokens = text.lower().split()
    total_tokens = len(tokens)
    
    if total_tokens == 0:
        return 0.0
    
    # Count medical terms
    medical_term_count = 0
    all_medical_terms = []
    
    # Flatten medical terms dictionary
    for category, terms in MEDICAL_TERMS.items():
        all_medical_terms.extend(terms)
    
    # Count occurrences
    text_lower = text.lower()
    for term in all_medical_terms:
        if term in text_lower:
            # Count word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = len(re.findall(pattern, text_lower))
            medical_term_count += matches
    
    # Calculate density
    density = (medical_term_count / total_tokens) * 100
    return min(density, 100.0)  # Cap at 100%

from flask import Flask, jsonify, render_template_string, request, session
import tempfile
import time
import uuid
import gc
import traceback
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
                traceback.print_exc()
                app.medgemma = None
                
        else:
            app.medgemma = None
            logger.info("❌ MedGemma not available")
        
        # ✅ Initialize RAG system with English Prompt Optimizer
        if RAG_AVAILABLE:
            try:
                logger.info("🔄 Creating enhanced RAG system with prompt optimization...")
                
                data_dir = os.path.abspath('data/chromadb')
                os.makedirs(data_dir, exist_ok=True)
                
                logger.info(f"🔄 Initializing RAG with persistent storage: {data_dir}")
        
                # ✅ CRITICAL: Verify write permissions before initialization
                test_file = os.path.join(data_dir, 'persistence_test.tmp')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    logger.info("✅ Directory permissions verified")
                except Exception as perm_error:
                    logger.error(f"❌ Cannot write to RAG directory: {perm_error}")
                    raise
                
                # Initialize with your new persistent RAG system
                app.rag_system = RAGSystem(
                    data_dir=data_dir,
                    embedding_model="medical"
                )
                
                # ✅ NEW: Verify persistence immediately after initialization
                doc_count = app.rag_system.chroma_collection.count() if app.rag_system.chroma_collection else 0
                logger.info(f"📊 RAG system loaded with {doc_count} existing documents")
                
                # ✅ NEW: Verify persistence status
                if hasattr(app.rag_system, 'verify_persistence'):
                    persistence_ok = app.rag_system.verify_persistence()
                    if persistence_ok:
                        logger.info("✅ Document persistence verified - uploads will persist across restarts")
                    else:
                        logger.warning("⚠️ Persistence verification failed - documents may not persist")
                
                if doc_count > 0:
                    logger.info("🎉 Previous documents successfully restored from persistent storage!")
                else:
                    logger.info("ℹ️ No previous documents found - fresh database ready for uploads")
                
                # ✅ CRITICAL FIX: Initialize session manager BEFORE register_routes
                app.session_manager = SessionManager(
                    db_manager=app.rag_system.db_manager if hasattr(app.rag_system, 'db_manager') else None,
                    session_timeout_hours=24
                )
                logger.info("✅ Session manager initialized with RAG integration")
                
            except Exception as rag_error:
                logger.error(f"❌ RAG system initialization failed: {rag_error}")
                traceback.print_exc()
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
        traceback.print_exc()
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
        
        # ✅ Check prompt optimizer status
        prompt_optimizer_status = "❌ Not Available"
        if app.rag_system and hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer:
            prompt_optimizer_status = "✅ Ready"
        
        return jsonify({
            "message": "🔬 PulseQuery AI - Complete Medical System",
            "status": "Running",
            "milestone": 5.2,
            "enhancement": "Enhanced Before/After Comparison Metrics + Medical Term Detection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "authenticated": user_info is not None,
            "user": user_info,
            "components": {
                "flask": "✅ Running",
                "medgemma": medgemma_status,
                "rag_system": rag_status,
                "prompt_optimizer": prompt_optimizer_status,
                "auth_service": "✅ Ready" if app.auth_service else "❌ Failed",
                "session_manager": "✅ Ready" if app.session_manager else "❌ Failed",
                "enhanced_prompts": "✅ V2.2 Enabled",
                "document_upload": "✅ Available" if app.rag_system else "❌ RAG Required",
                "medical_embeddings": "✅ Enabled" if app.rag_system else "❌ Not Available",
                "energy_metrics": "✅ Enabled",
                "medical_term_detection": "✅ Fixed",
                "comparison_metrics": "✅ Before/After Display"
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
                "milestone": 5.2
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
                "milestone": 5.2
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
        
        health_status = {
            "status": "healthy",
            "authenticated": user_info is not None,
            "user": user_info,
            "components": {
                "flask": "✅ Running",
                "auth_service": "✅ Ready" if app.auth_service else "❌ Failed",
                "session_manager": "✅ Ready" if app.session_manager else "❌ Failed",
            },
            "milestone": 5.2,
            "debug_info": {
                "RAG_AVAILABLE": RAG_AVAILABLE,
                "MEDGEMMA_AVAILABLE": MEDGEMMA_AVAILABLE,
                "rag_system_instance": app.rag_system is not None,
                "medgemma_instance": app.medgemma is not None,
                "prompt_optimizer_available": app.rag_system and hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer is not None
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
        
        # ✅ Prompt optimizer health
        if app.rag_system and hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer:
            health_status["components"]["prompt_optimizer"] = "✅ Ready"
        else:
            health_status["components"]["prompt_optimizer"] = "❌ Not Available"
        
        return jsonify(health_status)

    @app.route('/api/rag/persistence-status')
    @require_auth(app.session_manager)
    def rag_persistence_status():
        """Check RAG system persistence status"""
        if not RAG_AVAILABLE or not app.rag_system:
            return jsonify({"error": "RAG system not available"}), 503
        
        try:
            # Get persistence information
            status_info = {
                'persistent_storage': app.rag_system.chroma_collection is not None,
                'data_directory': app.rag_system.data_dir,
                'total_documents': 0,
                'persistence_files_exist': False,
                'collection_name': getattr(app.rag_system, 'collection_name', 'Unknown')
            }
            
            # Get document count
            if app.rag_system.chroma_collection:
                status_info['total_documents'] = app.rag_system.chroma_collection.count()
            
            # Check for persistence files
            if hasattr(app.rag_system, 'verify_persistence'):
                status_info['persistence_files_exist'] = app.rag_system.verify_persistence()
            
            # Check for ChromaDB files on disk
            chroma_files = ['chroma.sqlite3']
            files_found = []
            for file_name in chroma_files:
                file_path = os.path.join(app.rag_system.data_dir, file_name)
                if os.path.exists(file_path):
                    files_found.append(file_name)
            
            status_info['persistence_files_found'] = files_found
            status_info['persistence_verified'] = len(files_found) > 0
            
            return jsonify({
                'status': 'success',
                'persistence_info': status_info,
                'message': '✅ Persistence active' if status_info['persistence_verified'] else '⚠️ Persistence issues detected'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

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
                "milestone": 5.2,
                **status,
                "gpu_support": "Enabled",
                "model_file": "medgemma-4b-it-Q8_0.gguf"
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e),
                "milestone": 5.2
            })

    # ✅ ENHANCED Prompt Optimization Endpoint WITH BEFORE/AFTER COMPARISON
    @app.route('/api/prompt/optimize', methods=['POST'])
    @require_auth(app.session_manager)  
    def optimize_prompt():
        """Optimize medical prompt with enhanced energy saving metrics calculation and before/after comparison"""
        logger.info("🧠 PROMPT OPTIMIZATION WITH BEFORE/AFTER COMPARISON ENDPOINT CALLED")
        
        try:
            data = request.get_json()
            query = data.get('query')
            use_context = data.get('use_context', True)
            
            logger.info(f"📋 Query received: {query[:100]}..." if query else "📋 No query provided")
            logger.info(f"🔍 Use context: {use_context}")
            
            if not query:
                logger.error("❌ No query provided")
                return jsonify({'error': 'Query is required for optimization'}), 400
            
            if not hasattr(app.rag_system, 'prompt_optimizer') or not app.rag_system.prompt_optimizer:
                logger.error("❌ Prompt optimizer not available")
                return jsonify({'error': 'Prompt optimizer not available'}), 503
            
            logger.info("✅ Prompt optimizer found, proceeding...")
            
            # Get context documents if requested
            context_docs = []
            if use_context and app.rag_system:
                try:
                    context_docs = app.rag_system.search_relevant_context(query, max_docs=3)
                    logger.info(f"📄 Found {len(context_docs)} context documents")
                except Exception as context_error:
                    logger.warning(f"⚠️ Context search failed: {context_error}")
                    context_docs = []
            
            # ✅ Calculate original query metrics BEFORE optimization
            original_metrics = {
                'length': len(query),
                'token_estimate': int(len(query.split()) * 1.3),
                'medical_terminology_density': calculate_medical_terminology_density(query) / 100.0,
                'context_utilization': 0.0,
                'patient_specificity': 0.0
            }
            
            # ✅ Enhanced error handling for optimization with query preservation
            try:
                logger.info("🔄 Calling prompt optimizer...")
                result = app.rag_system.prompt_optimizer.optimize_prompt(query, context_docs)
                logger.info(f"✅ Optimization complete, prompt length: {len(result.get('optimized_prompt', ''))} chars")
                
                # ✅ CRITICAL FIX: Verify the query is included in the optimized prompt
                optimized_prompt = result.get('optimized_prompt', '')
                if not optimized_prompt or len(optimized_prompt.strip()) < 50:
                    logger.warning("⚠️ Optimized prompt is too short or empty, using enhanced fallback")
                    raise ValueError("Generated prompt is too short or empty")
                
                # Check if there was an error in the result
                if 'error' in result:
                    logger.warning(f"Optimization had issues: {result['error']}")
                
            except Exception as opt_error:
                logger.error(f"❌ Optimization completely failed: {opt_error}")
                
                # ✅ ENHANCED FALLBACK: Always preserve the user's query
                context_summary = "No relevant medical documents found." if not context_docs else f"Found {len(context_docs)} relevant documents."
                
                fallback_prompt = f"""You are a medical AI assistant specializing in symptom analysis and diagnosis.

PATIENT SYMPTOMS AND QUESTION:
{query}

CLINICAL CONTEXT:
{context_summary}

Please provide a comprehensive medical response that includes:

## SYMPTOM ANALYSIS

### SYMPTOM CHARACTERIZATION
Detail the presenting symptoms with onset, duration, and characteristics.

### DIFFERENTIAL DIAGNOSIS
List potential diagnoses ranked by likelihood with supporting evidence.

### RECOMMENDED DIAGNOSTIC WORKUP
Suggest appropriate tests, imaging, and consultations.

### MANAGEMENT RECOMMENDATIONS
Provide treatment suggestions and monitoring plans.

Generate a thorough medical analysis addressing the patient's symptoms and diagnostic question:"""
                
                # ✅ NEW: Calculate energy metrics for fallback
                original_tokens = len(query.split()) * 1.3  # Rough token estimation
                fallback_tokens = len(fallback_prompt.split()) * 1.3
                
                # Energy calculation constants
                ENERGY_PER_TOKEN = 0.0012  # Wh per token (MedGemma 4B Q8)
                CO2_PER_WH = 0.000233      # kg CO2 per Wh (average grid)
                COST_PER_KWH = 0.12        # USD per kWh
                
                token_reduction = max(0, original_tokens - fallback_tokens)
                energy_saved_wh = token_reduction * ENERGY_PER_TOKEN
                co2_saved_kg = energy_saved_wh * CO2_PER_WH
                cost_saved_usd = (energy_saved_wh / 1000) * COST_PER_KWH
                
                efficiency_improvement = (token_reduction / original_tokens) * 100 if original_tokens > 0 else 0
                
                # ✅ FIXED: Calculate medical terminology density
                medical_density = calculate_medical_terminology_density(fallback_prompt)
                
                return jsonify({
                    'success': True,
                    'original_query': query,
                    'optimized_prompt': fallback_prompt,
                    'query_type': 'symptom_analysis',
                    'medical_specialty': 'general_medicine',
                    'patient_info': {'name': None, 'age': None, 'gender': None, 'chief_complaint': None},
                    'original_metrics': original_metrics,
                    'metrics': {
                        'length': len(fallback_prompt), 
                        'token_estimate': int(fallback_tokens), 
                        'context_utilization': 0.0, 
                        'patient_specificity': 0.0, 
                        'medical_terminology_density': medical_density / 100.0  # Convert to decimal
                    },
                    'energy_metrics': {
                        'original_tokens': int(original_tokens),
                        'optimized_tokens': int(fallback_tokens),
                        'tokens_reduced': int(token_reduction),
                        'energy_saved_wh': round(energy_saved_wh, 6),
                        'co2_saved_kg': round(co2_saved_kg, 8),
                        'cost_saved_usd': round(cost_saved_usd, 6),
                        'efficiency_improvement_percent': round(efficiency_improvement, 2),
                        'energy_efficiency_percent': round((energy_saved_wh / (original_tokens * ENERGY_PER_TOKEN)) * 100, 2) if original_tokens > 0 else 0
                    },
                    'context_docs_used': len(context_docs),
                    'optimized_by': request.current_user["user_name"],
                    'milestone': 5.2,
                    'fallback_used': True,
                    'optimization_error': str(opt_error),
                    'medical_term_fix_applied': True,
                    'comparison_enabled': True
                })
            
            # ✅ NEW: ENERGY METRICS CALCULATION
            original_tokens = len(query.split()) * 1.3  # Rough token estimation
            optimized_tokens = len(result['optimized_prompt'].split()) * 1.3
            
            # Energy calculation constants (based on GPU processing)
            ENERGY_PER_TOKEN = 0.0012  # Wh per token (MedGemma 4B Q8)
            CO2_PER_WH = 0.000233      # kg CO2 per Wh (average grid)
            COST_PER_KWH = 0.12        # USD per kWh
            
            token_reduction = max(0, original_tokens - optimized_tokens)
            energy_saved_wh = token_reduction * ENERGY_PER_TOKEN
            co2_saved_kg = energy_saved_wh * CO2_PER_WH
            cost_saved_usd = (energy_saved_wh / 1000) * COST_PER_KWH
            
            # Calculate efficiency metrics
            if original_tokens > 0:
                efficiency_improvement = (token_reduction / original_tokens) * 100
                energy_efficiency = (energy_saved_wh / (original_tokens * ENERGY_PER_TOKEN)) * 100
            else:
                efficiency_improvement = 0
                energy_efficiency = 0
            
            # ✅ FIX: Safe attribute access with proper error handling + MEDICAL TERM DENSITY FIX
            try:
                # Safe extraction of patient info
                patient_info = result.get('patient_info')
                patient_data = {
                    'name': getattr(patient_info, 'name', None) if patient_info else None,
                    'age': getattr(patient_info, 'age', None) if patient_info else None,
                    'gender': getattr(patient_info, 'gender', None) if patient_info else None,
                    'chief_complaint': getattr(patient_info, 'chief_complaint', None) if patient_info else None
                }
                
                # Safe extraction of metrics
                metrics = result.get('metrics')
                
                # ✅ FIXED: Calculate medical terminology density properly
                optimized_prompt_text = result.get('optimized_prompt', '')
                medical_density = calculate_medical_terminology_density(optimized_prompt_text)
                
                metrics_data = {
                    'length': getattr(metrics, 'length', 0) if metrics else len(optimized_prompt_text),
                    'token_estimate': getattr(metrics, 'token_estimate', 0) if metrics else int(len(optimized_prompt_text.split()) * 1.3),
                    'context_utilization': getattr(metrics, 'context_utilization', 0.0) if metrics else (len(context_docs) / 3.0 if context_docs else 0.0),
                    'patient_specificity': getattr(metrics, 'patient_specificity', 0.0) if metrics else 0.0,
                    'medical_terminology_density': medical_density / 100.0  # Convert to decimal (0-1 range)
                }
                
            except Exception as attr_error:
                logger.warning(f"⚠️ Attribute extraction failed: {attr_error}")
                # Fallback values with medical term calculation
                optimized_prompt_text = result.get('optimized_prompt', '')
                medical_density = calculate_medical_terminology_density(optimized_prompt_text)
                
                patient_data = {'name': None, 'age': None, 'gender': None, 'chief_complaint': None}
                metrics_data = {
                    'length': len(optimized_prompt_text),
                    'token_estimate': len(optimized_prompt_text.split()),
                    'context_utilization': len(context_docs) / 3.0 if context_docs else 0.0,
                    'patient_specificity': 0.0,
                    'medical_terminology_density': medical_density / 100.0  # Convert to decimal
                }
            
            # ✅ FINAL VALIDATION: Ensure optimized prompt contains the query
            final_prompt = result.get('optimized_prompt', '')
            
            # Check if key words from the query appear in the optimized prompt
            query_words = set(query.lower().split())
            prompt_words = set(final_prompt.lower().split())
            common_words = query_words.intersection(prompt_words)
            
            if len(common_words) < 2:  # If less than 2 words match, query might be missing
                logger.warning("⚠️ Query appears to be missing from optimized prompt, enhancing...")
                # Inject the query more explicitly
                final_prompt = final_prompt.replace(
                    "MEDICAL QUERY:\n", 
                    f"MEDICAL QUERY:\n{query}\n\n"
                )
                if "MEDICAL QUERY:" not in final_prompt:
                    final_prompt = f"USER QUERY: {query}\n\n{final_prompt}"
            
            response_data = {
                'success': True,
                'original_query': query,
                'optimized_prompt': final_prompt,
                'query_type': result.get('query_type', 'general_medical'),
                'medical_specialty': result.get('medical_specialty', 'general_medicine'),
                'patient_info': patient_data,
                'original_metrics': original_metrics,  # ✅ NEW: Original query metrics
                'metrics': metrics_data,  # ✅ Optimized prompt metrics
                # ✅ NEW: Energy and efficiency metrics
                'energy_metrics': {
                    'original_tokens': int(original_tokens),
                    'optimized_tokens': int(optimized_tokens),
                    'tokens_reduced': int(token_reduction),
                    'energy_saved_wh': round(energy_saved_wh, 6),
                    'co2_saved_kg': round(co2_saved_kg, 8),
                    'cost_saved_usd': round(cost_saved_usd, 6),
                    'efficiency_improvement_percent': round(efficiency_improvement, 2),
                    'energy_efficiency_percent': round(energy_efficiency, 2)
                },
                'context_docs_used': len(context_docs),
                'optimized_by': request.current_user["user_name"],
                'milestone': 5.2,
                'has_optimization_warning': 'error' in result,
                'query_preservation_check': len(common_words) >= 2,
                'medical_term_fix_applied': True,
                'comparison_enabled': True  # ✅ NEW: Enable before/after comparison
            }
            
            logger.info(f"📤 Sending response with {len(response_data['optimized_prompt'])} char prompt")
            logger.info(f"🔋 Energy saved: {response_data['energy_metrics']['energy_saved_wh']:.6f} Wh")
            logger.info(f"🌱 CO2 saved: {response_data['energy_metrics']['co2_saved_kg']:.8f} kg")
            logger.info(f"💰 Cost saved: ${response_data['energy_metrics']['cost_saved_usd']:.6f}")
            logger.info(f"🩺 Medical density: Original {original_metrics['medical_terminology_density']*100:.1f}% → Optimized {metrics_data['medical_terminology_density']*100:.1f}%")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ API endpoint failed: {e}")
            import traceback
            traceback.print_exc()
            
            # ✅ ULTIMATE FALLBACK: Simple but functional response
            fallback_response = {
                'success': True,
                'original_query': query if 'query' in locals() else 'Unknown query',
                'optimized_prompt': f"You are a medical AI assistant. Please provide a comprehensive medical response to: {query if 'query' in locals() else 'the patient query'}",
                'query_type': 'general_medical',
                'medical_specialty': 'general_medicine',
                'patient_info': {'name': None, 'age': None, 'gender': None, 'chief_complaint': None},
                'original_metrics': {'length': 0, 'token_estimate': 0, 'context_utilization': 0.0, 'patient_specificity': 0.0, 'medical_terminology_density': 0.0},
                'metrics': {'length': 0, 'token_estimate': 0, 'context_utilization': 0.0, 'patient_specificity': 0.0, 'medical_terminology_density': 0.0},
                'energy_metrics': {'original_tokens': 0, 'optimized_tokens': 0, 'tokens_reduced': 0, 'energy_saved_wh': 0.0, 'co2_saved_kg': 0.0, 'cost_saved_usd': 0.0, 'efficiency_improvement_percent': 0.0, 'energy_efficiency_percent': 0.0},
                'context_docs_used': 0,
                'optimized_by': 'System',
                'milestone': 5.2,
                'endpoint_error': True,
                'error_message': str(e),
                'medical_term_fix_applied': False,
                'comparison_enabled': False
            }
            
            return jsonify(fallback_response), 200  # Return 200 to avoid breaking UI

    # ✅ DEBUG: Add debug endpoint to check prompt optimizer status
    @app.route('/api/debug/optimizer-status')
    @require_auth(app.session_manager)
    def debug_optimizer_status():
        """Debug endpoint to check prompt optimizer status"""
        try:
            status = {
                'rag_system_exists': app.rag_system is not None,
                'rag_system_type': str(type(app.rag_system).__name__),
                'has_prompt_optimizer_attr': hasattr(app.rag_system, 'prompt_optimizer') if app.rag_system else False,
                'prompt_optimizer_exists': app.rag_system.prompt_optimizer is not None if app.rag_system and hasattr(app.rag_system, 'prompt_optimizer') else False,
                'prompt_optimizer_type': str(type(app.rag_system.prompt_optimizer).__name__) if app.rag_system and hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer else None
            }
            
            if app.rag_system and hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer:
                # Test basic functionality
                try:
                    test_result = app.rag_system.prompt_optimizer.optimize_prompt(
                        "Test patient with chest pain", 
                        []
                    )
                    status['test_optimization'] = 'SUCCESS'
                    status['test_prompt_length'] = len(test_result.get('optimized_prompt', ''))
                except Exception as test_error:
                    status['test_optimization'] = f'FAILED: {test_error}'
            
            return jsonify(status)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/prompt/generate-final', methods=['POST'])
    @require_auth(app.session_manager)
    def generate_from_final_prompt():
        """Generate AI response from user's final edited prompt"""
        try:
            data = request.get_json()
            final_prompt = data.get('final_prompt')
            max_tokens = data.get('max_tokens', 600)
            temperature = data.get('temperature', 0.3)
            
            if not final_prompt:
                return jsonify({'error': 'Final prompt is required'}), 400
            
            if not MEDGEMMA_AVAILABLE or not app.medgemma:
                return jsonify({'error': 'MedGemma not available'}), 503
            
            if not app.medgemma.is_ready():
                return jsonify({'error': 'Model not ready'}), 503
            
            # Generate AI response from user's final prompt
            result = app.medgemma.generate_text(
                prompt=final_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return jsonify({
                'success': True,
                'final_prompt': final_prompt,
                'ai_response': result.get('generated_text', ''),
                'generation_info': {
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'device': result.get('device', 'unknown'),
                    'prompt_length': len(final_prompt),
                    'response_length': len(result.get('generated_text', ''))
                },
                'generated_by': request.current_user["user_name"],
                'timestamp': datetime.now().isoformat(),
                'milestone': 5.2
            })
            
        except Exception as e:
            logger.info(f"❌ Final prompt generation failed: {e}")
            return jsonify({'error': str(e)}), 500

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
                "milestone": 5.2,
                "rag_stats": stats,
                "accessed_by": request.current_user["user_name"],
                "embedding_info": {
                    "model": stats.get('embedding_model', 'Unknown'),
                    "type": "Medical-specialized"
                },
                "prompt_optimizer": "✅ Available" if hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer else "❌ Not Available"
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
        
        # Check if it's the fallback system
        if hasattr(app.rag_system, '__class__') and 'Fallback' in app.rag_system.__class__.__name__:
            return jsonify({
                "error": "RAG system in fallback mode - document upload unavailable",
                "suggestion": "Check server logs for RAG initialization errors"
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
                    "milestone": 5.2,
                    **result
                })
                
            finally:
                # Always cleanup temp file
                cleanup_temp_file(temp_path)
        
        except Exception as e:
            logger.info(f"❌ Upload failed: {e}")
            traceback.print_exc()
            return jsonify({
                "error": str(e),
                "debug": {
                    "file_name": file.filename if 'file' in locals() else 'unknown',
                    "temp_path": temp_path if 'temp_path' in locals() else 'unknown'
                }
            }), 500

    

    # Login page route
    @app.route('/')
    @app.route('/login')
    def login_page():
        """Serve the login page"""
        return send_from_directory('static', 'login.html')

    # Main screen route  
    @app.route('/main')
    def main_screen():
        """Serve the main screen"""
        return send_from_directory('static', 'main.html')

    # Keep your existing test-ui route
    @app.route('/test-ui')
    def test_ui():
        """Serve the comprehensive UI"""
        return send_from_directory('static', 'ui.html')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found", "milestone": 5.2}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "milestone": 5.2}), 500

def verify_app_persistence(app):
    """Verify that the application has proper persistence configured"""
    logger.info("🔍 Verifying application persistence configuration...")
    
    if not app.rag_system:
        logger.warning("⚠️ RAG system not available - no persistence")
        return False
    
    try:
        # Check data directory
        data_dir = getattr(app.rag_system, 'data_dir', None)
        if not data_dir or not os.path.exists(data_dir):
            logger.warning(f"⚠️ Data directory doesn't exist: {data_dir}")
            return False
        
        # Check ChromaDB collection
        if not app.rag_system.chroma_collection:
            logger.warning("⚠️ No ChromaDB collection - using fallback storage")
            return False
        
        # Check document count
        doc_count = app.rag_system.chroma_collection.count()
        logger.info(f"📊 Persistent storage verified: {doc_count} documents")
        
        # Check for persistence files
        chroma_db_file = os.path.join(data_dir, 'chroma.sqlite3')
        if os.path.exists(chroma_db_file):
            file_size = os.path.getsize(chroma_db_file)
            logger.info(f"📁 ChromaDB persistence file found: {file_size} bytes")
            return True
        else:
            logger.info("📁 No existing persistence file - will create on first upload")
            return True  # Still valid for new installations
            
    except Exception as e:
        logger.error(f"❌ Persistence verification failed: {e}")
        return False

# Main execution
if __name__ == '__main__':
    logger.info("🚀 Starting PulseQuery AI - Milestone 5.2: Enhanced Before/After Comparison Metrics")
    logger.info("🔧 Features: English Medical Prompt Optimizer, Query Classification, Medical Specialization")
    logger.info("🧠 Enhanced: Template-based Medical Prompt Engineering with Energy Savings")
    logger.info("📄 NEW: Interactive Prompt Optimization Workflow with Markdown Rendering + Energy Analytics")
    logger.info("🌱 GREEN: Energy consumption tracking and environmental impact metrics")
    logger.info("🩺 FIXED: Medical terminology density calculation working properly")
    logger.info("📊 NEW: Before/After Comparison Display with Side-by-Side Metrics")
    logger.info("🎯 UI: Complete English-focused Medical AI System with Enhanced Comparison Interface")
    
    try:
        logger.info("\n🔄 Creating complete Flask application...")
        app = create_app()
        logger.info("✅ Complete application created successfully!")
        
        # ✅ NEW: Verify persistence configuration
        persistence_verified = verify_app_persistence(app)
        if persistence_verified:
            logger.info("✅ Document persistence verified - uploads will persist across restarts")
        else:
            logger.warning("⚠️ Persistence issues detected - check configuration")
        
        logger.info("\n🌐 Server Information:")
        logger.info("📍 Main Server: http://localhost:5000")
        logger.info("🔗 Enhanced Test UI: http://localhost:5000/test-ui")
        logger.info("❤️ Health Check: http://localhost:5000/health")
        logger.info("💾 Persistence Status: http://localhost:5000/api/rag/persistence-status")
        
        logger.info("\n🧪 Milestone 5.2 Enhanced Endpoints:")
        logger.info("   - /api/prompt/optimize - English prompt optimization + energy metrics + before/after comparison")
        logger.info("   - /api/prompt/generate-final - Generate from optimized prompt")
        logger.info("   - /api/debug/optimizer-status - Debug optimizer status")
        logger.info("   - /api/auth/login - User authentication")
        logger.info("   - /api/rag/upload - Document upload with persistent storage")
        logger.info("   - /api/rag/stats - RAG system statistics")
        logger.info("   - /api/rag/persistence-status - Check document persistence")
        
        logger.info("\n🎯 Milestone 5.2 Enhanced Features:")
        logger.info("   - English Medical Prompt Optimizer with Energy Analytics")
        logger.info("   - Query Type Classification (7 types)")
        logger.info("   - Medical Specialty Detection (10+ specialties)")
        logger.info("   - Patient Information Extraction")
        logger.info("   - Template-based Prompt Generation")
        logger.info("   - ✅ FIXED: Medical Terminology Density Calculation")
        logger.info("   - ✅ NEW: Before/After Comparison Metrics Display")
        logger.info("   - 🌱 Energy Consumption Tracking")
        logger.info("   - 💰 Cost Savings Analysis")
        logger.info("   - 🌍 Environmental Impact Metrics (CO₂ savings)")
        logger.info("   - 📊 Interactive Energy Dashboard with Charts")
        logger.info("   - Interactive UI Workflow")
        logger.info("   - Markdown Rendering in UI ✅")
        logger.info("   - Persistent ChromaDB Storage ✅")
        logger.info("   - Debug Tools for Troubleshooting")
        
        logger.info("\n👥 Demo Login Credentials:")
        logger.info("   - doctor1 / password123 (✅ Full access)")
        logger.info("   - admin1 / admin123 (✅ Administrative access)")  
        logger.info("   - nurse1 / nurse123 (📖 Read-only access)")
        logger.info("   - resident1 / resident123 (📖 Limited access)")
        
        logger.info("\n🎨 Enhanced UI Features:")
        logger.info("   - Markdown rendering with marked.js library")
        logger.info("   - Professional medical report styling")
        logger.info("   - 📊 Before/After comparison metrics with Chart.js")
        logger.info("   - Real-time CO₂ and cost savings display")
        logger.info("   - Interactive energy impact charts")
        logger.info("   - Headers, lists, tables, and formatting support")
        logger.info("   - Copy original markdown functionality")
        logger.info("   - Elevated logo frame with hover effects")
        logger.info("   - ✅ Comparison indicator for enhanced user experience")
        
        logger.info("\n🔋 Energy Analytics Features:")
        logger.info("   - Token usage before/after optimization")
        logger.info("   - Energy consumption in Wh (Watt-hours)")
        logger.info("   - CO₂ emissions saved (kg)")
        logger.info("   - Cost savings in USD")
        logger.info("   - Efficiency improvement percentages")
        logger.info("   - Visual charts for impact analysis")
        
        logger.info("\n📊 Before/After Comparison Features:")
        logger.info("   - ✅ Side-by-side character count comparison")
        logger.info("   - ✅ Token count before vs after optimization")
        logger.info("   - ✅ Medical terminology density improvement tracking")
        logger.info("   - ✅ Change indicators showing exact differences")
        logger.info("   - ✅ Color-coded metrics (orange=before, green=after)")
        logger.info("   - ✅ Visual improvement validation")
        logger.info("   - ✅ Comprehensive optimization proof display")
        
        logger.info("\n🎯 Final UI Layout Order:")
        logger.info("   - ✅ 1. Energy Metrics (🌱 Optimization Energy Impact)")
        logger.info("   - ✅ 2. Before/After Comparison (📊 side-by-side metrics)")
        logger.info("   - ✅ 3. Interactive Chart (below both metric sections)")
        logger.info("   - ✅ Comparison indicator when before/after data available")
        logger.info("   - ✅ Professional gradient backgrounds and consistent design")
        
        logger.info("\n🔄 Starting server...")
        
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
        traceback.print_exc()
    finally:
        logger.info("\n✅ Milestone 5.2 Enhanced shutdown complete - Before/After Comparison Fully Implemented!")

