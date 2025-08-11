import logging
import warnings
from datetime import datetime, timezone
import os

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

from flask import Flask, jsonify, render_template_string, request, session
import tempfile
import time
import uuid
import gc
import re
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
            "milestone": 5,
            "enhancement": "English Prompt Optimization System with Energy Metrics",
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
                "enhanced_prompts": "✅ V2.0 Enabled",
                "document_upload": "✅ Available" if app.rag_system else "❌ RAG Required",
                "medical_embeddings": "✅ Enabled" if app.rag_system else "❌ Not Available",
                "energy_metrics": "✅ Enabled"
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
                "milestone": 5
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
                "milestone": 5
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
            "milestone": 5,
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
                "milestone": 5,
                **status,
                "gpu_support": "Enabled",
                "model_file": "medgemma-4b-it-Q8_0.gguf"
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e),
                "milestone": 5
            })

    # ✅ ENHANCED Prompt Optimization Endpoint WITH ENERGY METRICS
    @app.route('/api/prompt/optimize', methods=['POST'])
    @require_auth(app.session_manager)  
    def optimize_prompt():
        """Optimize medical prompt with enhanced energy saving metrics calculation"""
        logger.info("🧠 PROMPT OPTIMIZATION WITH ENERGY METRICS ENDPOINT CALLED")
        
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
                
                return jsonify({
                    'success': True,
                    'original_query': query,
                    'optimized_prompt': fallback_prompt,
                    'query_type': 'symptom_analysis',
                    'medical_specialty': 'general_medicine',
                    'patient_info': {'name': None, 'age': None, 'gender': None, 'chief_complaint': None},
                    'metrics': {
                        'length': len(fallback_prompt), 
                        'token_estimate': int(fallback_tokens), 
                        'context_utilization': 0.0, 
                        'patient_specificity': 0.0, 
                        'medical_terminology_density': 0.3
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
                    'milestone': 5,
                    'fallback_used': True,
                    'optimization_error': str(opt_error)
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
            
            # ✅ FIX: Safe attribute access with proper error handling
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
                metrics_data = {
                    'length': getattr(metrics, 'length', 0) if metrics else 0,
                    'token_estimate': getattr(metrics, 'token_estimate', 0) if metrics else 0,
                    'context_utilization': getattr(metrics, 'context_utilization', 0.0) if metrics else 0.0,
                    'patient_specificity': getattr(metrics, 'patient_specificity', 0.0) if metrics else 0.0,
                    'medical_terminology_density': getattr(metrics, 'medical_terminology_density', 0.0) if metrics else 0.0
                }
                
            except Exception as attr_error:
                logger.warning(f"⚠️ Attribute extraction failed: {attr_error}")
                # Fallback values
                patient_data = {'name': None, 'age': None, 'gender': None, 'chief_complaint': None}
                metrics_data = {
                    'length': len(result.get('optimized_prompt', '')),
                    'token_estimate': len(result.get('optimized_prompt', ''))//4,
                    'context_utilization': 0.0,
                    'patient_specificity': 0.0,
                    'medical_terminology_density': 0.0
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
                'metrics': metrics_data,
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
                'milestone': 5,
                'has_optimization_warning': 'error' in result,
                'query_preservation_check': len(common_words) >= 2
            }
            
            logger.info(f"📤 Sending response with {len(response_data['optimized_prompt'])} char prompt")
            logger.info(f"🔋 Energy saved: {response_data['energy_metrics']['energy_saved_wh']:.6f} Wh")
            logger.info(f"🌱 CO2 saved: {response_data['energy_metrics']['co2_saved_kg']:.8f} kg")
            logger.info(f"💰 Cost saved: ${response_data['energy_metrics']['cost_saved_usd']:.6f}")
            
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
                'metrics': {
                    'length': 0,
                    'token_estimate': 0,
                    'context_utilization': 0.0,
                    'patient_specificity': 0.0,
                    'medical_terminology_density': 0.0
                },
                'energy_metrics': {
                    'original_tokens': 0,
                    'optimized_tokens': 0,
                    'tokens_reduced': 0,
                    'energy_saved_wh': 0.0,
                    'co2_saved_kg': 0.0,
                    'cost_saved_usd': 0.0,
                    'efficiency_improvement_percent': 0.0,
                    'energy_efficiency_percent': 0.0
                },
                'context_docs_used': 0,
                'optimized_by': 'System',
                'milestone': 5,
                'endpoint_error': True,
                'error_message': str(e)
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
                'milestone': 5
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
                "milestone": 5,
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
                    "milestone": 5,
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

    # ✅ Enhanced Test UI with ENERGY METRICS DISPLAY
    @app.route('/test-ui')
    def test_ui():
        """Complete test interface with energy metrics display"""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>PulseQuery AI - Prompt Optimization with Energy Metrics</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/marked@4.3.0/marked.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 20px; 
                    background-color: #f8f9fa; 
                }
                .container { 
                    max-width: 1400px; 
                    margin: 0 auto; 
                }
                .card { 
                    border: 1px solid #dee2e6; 
                    padding: 20px; 
                    margin: 15px 0; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
                    background: white; 
                }
                .card-header { 
                    background: linear-gradient(135deg, #007bff, #0056b3); 
                    color: white; 
                    padding: 15px; 
                    margin: -20px -20px 20px -20px; 
                    border-radius: 8px 8px 0 0; 
                }
                .btn { 
                    padding: 10px 20px; 
                    border: none; 
                    cursor: pointer; 
                    margin: 5px; 
                    border-radius: 5px; 
                    font-weight: 500; 
                    transition: all 0.2s; 
                }
                .btn-success { background: #28a745; color: white; } 
                .btn-success:hover { background: #218838; }
                .btn-warning { background: #ffc107; color: #856404; } 
                .btn-warning:hover { background: #e0a800; }
                .btn-danger { background: #dc3545; color: white; } 
                .btn-danger:hover { background: #c82333; }
                .btn-info { background: #17a2b8; color: white; } 
                .btn-info:hover { background: #138496; }
                .btn-primary { background: #007bff; color: white; } 
                .btn-primary:hover { background: #0056b3; }
                .btn-secondary { background: #6c757d; color: white; }
                .btn-secondary:hover { background: #545b62; }
                .alert { 
                    padding: 15px; 
                    margin: 15px 0; 
                    border-radius: 5px; 
                    border: 1px solid transparent; 
                }
                .alert-success { background: #d4edda; border-color: #c3e6cb; color: #155724; }
                .alert-info { background: #d1ecf1; border-color: #bee5eb; color: #0c5460; }
                .alert-warning { background: #fff3cd; border-color: #ffeaa7; color: #856404; }
                .alert-danger { background: #f8d7da; border-color: #f5c6cb; color: #721c24; }
                .form-control, .form-select { 
                    border-radius: 5px; 
                    border: 1px solid #ced4da; 
                    padding: 8px 12px; 
                }
                .status-badge { 
                    padding: 4px 8px; 
                    border-radius: 12px; 
                    font-size: 12px; 
                    font-weight: bold; 
                }
                .status-ready { background: #d4edda; color: #155724; }
                .status-loading { background: #fff3cd; color: #856404; }
                .status-error { background: #f8d7da; color: #721c24; }
                .optimization-info {
                    background: #e7f3ff;
                    border: 1px solid #b8daff;
                    border-radius: 4px;
                    padding: 8px;
                    margin: 8px 0;
                    font-size: 0.9em;
                }
                
                /* ✅ UPDATED: Prompt Metrics styling to match Energy Metrics */
                .prompt-metrics {
                    background: linear-gradient(135deg, #e8f5e8, #f0f9ff);
                    border: 1px solid #4caf50;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 16px 0;
                }

                .prompt-metric {
                    background: white;
                    border: 1px solid #e1f5fe;
                    border-radius: 6px;
                    padding: 10px;
                    margin: 5px;
                    text-align: center;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                }

                .prompt-metric-value {
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: #1976d2;
                }

                .prompt-metric-label {
                    font-size: 0.8rem;
                    color: #666;
                    margin-top: 4px;
                }

                .prompt-metrics h6 {
                    color: #2e7d32;
                    margin-bottom: 16px;
                    font-weight: 600;
                }
                
                /* ✅ NEW: Energy Metrics Styling */
                .energy-metrics {
                    background: linear-gradient(135deg, #e8f5e8, #f0f9ff);
                    border: 1px solid #4caf50;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 16px 0;
                }
                .energy-metric {
                    background: white;
                    border: 1px solid #e1f5fe;
                    border-radius: 6px;
                    padding: 10px;
                    margin: 5px;
                    text-align: center;
                }
                .energy-value {
                    font-size: 1.2rem;
                    font-weight: 600;
                    color: #1976d2;
                }
                .energy-label {
                    font-size: 0.8rem;
                    color: #666;
                }
                .environmental-impact {
                    background: rgba(76, 175, 80, 0.1);
                    border-left: 4px solid #4caf50;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 4px;
                }
                
                .loading-spinner {
                    display: inline-block;
                    width: 16px;
                    height: 16px;
                    border: 2px solid #f3f3f3;
                    border-top: 2px solid #007bff;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                    margin-right: 8px;
                }
                .header-container {
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    padding: 20px 24px;
                    background: transparent;
                }

                .logo-frame {
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
                    padding: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    min-width: 80px;
                    min-height: 80px;
                    transition: transform 0.2s ease, box-shadow 0.2s ease;
                }

                .logo-frame:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
                }

                .logo-image {
                    width: 60px;
                    height: 60px;
                    object-fit: contain;
                    border-radius: 8px;
                }

                .header-content {
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }

                .main-title {
                    margin: 0;
                    font-size: 2.2rem;
                    font-weight: 700;
                    color: white;
                    line-height: 1.1;
                }

                .subtitle {
                    margin: 4px 0 0 0;
                    font-size: 0.95rem;
                    color: rgba(255, 255, 255, 0.9);
                    font-weight: 400;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                /* ✅ MARKDOWN RENDERING STYLES */
                #aiResponse {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                    line-height: 1.6;
                    color: #333;
                }
                #aiResponse h1, #aiResponse h2, #aiResponse h3, #aiResponse h4, #aiResponse h5, #aiResponse h6 {
                    color: #2c3e50;
                    margin-top: 1.5em;
                    margin-bottom: 0.5em;
                    font-weight: 600;
                }
                #aiResponse h1 { font-size: 1.8em; border-bottom: 2px solid #3498db; padding-bottom: 0.3em; }
                #aiResponse h2 { font-size: 1.5em; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.3em; }
                #aiResponse h3 { font-size: 1.3em; color: #34495e; }
                #aiResponse h4 { font-size: 1.1em; color: #34495e; }
                #aiResponse ul, #aiResponse ol {
                    margin: 1em 0;
                    padding-left: 2em;
                }
                #aiResponse li {
                    margin: 0.5em 0;
                }
                #aiResponse p {
                    margin: 1em 0;
                    text-align: justify;
                }
                #aiResponse strong {
                    color: #2c3e50;
                    font-weight: 600;
                }
                #aiResponse em {
                    color: #7f8c8d;
                    font-style: italic;
                }
                #aiResponse blockquote {
                    border-left: 4px solid #3498db;
                    margin: 1.5em 0;
                    padding: 0.5em 0 0.5em 1em;
                    background-color: #ecf0f1;
                    font-style: italic;
                }
                #aiResponse code {
                    background-color: #f8f9fa;
                    padding: 0.2em 0.4em;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    color: #e74c3c;
                }
                #aiResponse pre {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    padding: 1em;
                    border-radius: 5px;
                    overflow-x: auto;
                    margin: 1.5em 0;
                }
                #aiResponse pre code {
                    background: none;
                    color: inherit;
                    padding: 0;
                }
                #aiResponse table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 1.5em 0;
                }
                #aiResponse th, #aiResponse td {
                    border: 1px solid #bdc3c7;
                    padding: 0.75em;
                    text-align: left;
                }
                #aiResponse th {
                    background-color: #ecf0f1;
                    font-weight: 600;
                    color: #2c3e50;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Header -->
                <div class="card">
                    <div class="card-header">
                        <div class="header-container">
                        <div class="logo-frame">
                            <img src="/static/logo.png" alt="PulseQuery Logo" class="logo-image">
                        </div>
                        <div class="header-content">
                            <h1 class="main-title">PulseQuery AI</h1>
                            <p class="subtitle">Advanced Medical Intelligence Platform with Energy Optimization</p>
                        </div>
                    </div>
                    </div>
                    <div class="row">
                        <div class="col-md-6">
                            <p><strong>Version:</strong> 5.0 - Clinical Intelligence Platform</p>
                            <p><strong>Status:</strong> <span id="systemStatus" class="status-badge">Checking...</span></p>
                        </div>
                        <div class="col-md-6">
                            <p><strong>Core Capabilities:</strong> Smart Query Analysis • Medical AI • Patient Data Extraction • Energy Optimization</p>
                            <p><strong>Latest:</strong> Enhanced Prompt Engineering + Energy Metrics + Markdown Support</p>
                        </div>
                    </div>
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

                <!-- ✅ AI-Assisted Medical Query Processing with Energy Metrics -->
                <div class="card" id="promptOptimizationCard" style="display: none;">
                    <div class="card-header">
                        <h5>🧠 AI-Assisted Prompt Optimization with Energy Metrics</h5>
                        <small>Submit medical query → AI optimizes prompt → Review & edit → Generate detailed insights → View energy savings</small>
                    </div>                    
                    <!-- Step 1: User Query Input -->
                    <div class="mb-4">
                        <label class="form-label"><strong>Step 1: Enter Medical Query</strong></label>
                        <textarea id="userQuery" class="form-control" rows="4" 
                                  placeholder="Enter your medical query here...&#10;&#10;Examples:&#10;• Patient Name: Rogers, Pamela, Age: 56, Chief Complaint: Chest pain and shortness of breath&#10;• 45-year-old male with diabetes presenting with foot ulcer&#10;• Patient with history of hypertension needs treatment plan review"></textarea>
                        <div class="mt-2">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="useContext" checked>
                                <label class="form-check-label" for="useContext">
                                    Use document context for optimization (recommended)
                                </label>
                            </div>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-primary btn-lg" onclick="optimizePrompt()">
                                🔍 Optimize Prompt with Energy Analysis
                            </button>
                        </div>
                    </div>
                    
                    <!-- Step 2: Optimized Prompt Display & Editing -->
                    <div id="optimizedPromptSection" style="display: none;">
                        <hr>
                        <label class="form-label"><strong>Step 2: Review & Edit Optimized Prompt</strong></label>
                        <div class="optimization-info" id="optimizationInfo"></div>
                        
                        <!-- ✅ NEW: Energy Metrics Display -->
                        <div id="energyMetricsSection" class="energy-metrics" style="display: none;">
                            <h6><i class="fas fa-leaf"></i> 🌱 Optimization Energy Impact</h6>
                            <div class="row" id="energyMetricsGrid">
                                <!-- Energy metrics will be populated here -->
                            </div>
                            <div class="environmental-impact" id="environmentalImpact">
                                <!-- Environmental impact summary -->
                            </div>
                            <canvas id="energyChart" width="400" height="150" style="margin-top: 10px;"></canvas>
                        </div>
                        
                        <textarea id="optimizedPrompt" class="form-control" rows="15" 
                                  placeholder="Optimized prompt will appear here..."></textarea>
                        
                        <div class="mt-3">
                            <div class="row">
                                <div class="col-md-8">
                                    <button class="btn btn-success btn-lg" onclick="generateFromPrompt()">
                                        🤖 Generate AI Response
                                    </button>
                                    <button class="btn btn-secondary" onclick="resetOptimization()">
                                        🔄 Start Over
                                    </button>
                                </div>
                                <div class="col-md-4">
                                    <!-- ✅ UPDATED: New prompt metrics display matching energy metrics style -->
                                    <div id="promptMetrics" class="prompt-metrics" style="display: none;">
                                        <h6><i class="fas fa-chart-line"></i> 📊 Prompt Quality Metrics</h6>
                                        <div class="row" id="promptMetricsGrid">
                                            <!-- Metrics cards will be populated here -->
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Step 3: AI Response Display -->
                    <div id="aiResponseSection" style="display: none;">
                        <hr>
                        <label class="form-label"><strong>Step 3: AI Generated Medical Response</strong></label>
                        <div id="aiResponse" class="alert alert-success"></div>
                        <div class="row">
                            <div class="col-md-8">
                                <div class="text-muted">
                                    <small id="generationInfo"></small>
                                </div>
                            </div>
                            <div class="col-md-4 text-end">
                                <button class="btn btn-info btn-sm" onclick="copyResponse()">📋 Copy Response</button>
                                <button class="btn btn-primary btn-sm" onclick="resetForNew()">✨ New Query</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Document Upload -->
                <div class="card" id="uploadCard" style="display: none;">
                    <div class="card-header">
                        <h5>📄 Medical Document Upload</h5>
                        <small>Upload patient records for AI analysis (Write permission required)</small>
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
                                </select>
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Language:</label>
                                <select class="form-select" id="docLanguage">
                                    <option value="en">🇺🇸 English</option>
                                </select>
                            </div>
                            <div class="col-md-2">
                                <label class="form-label">&nbsp;</label>
                                <button type="submit" class="btn btn-primary w-100">📤 Upload</button>
                            </div>
                        </div>
                    </form>
                    <div id="uploadResult" class="mt-3" style="display: none;"></div>
                </div>

                <!-- Footer -->
                <div class="card">
                    <div class="text-center">
                        <p class="mb-2"><strong>PulseQuery AI v5.0</strong></p>
                        <p class="mb-0 text-muted">Empowering Healthcare with AI-Driven Insights & Energy Optimization</p>
                    </div>
                </div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
                let currentUser = null;
                let sessionId = null;

                document.addEventListener('DOMContentLoaded', function() {
                    console.log('🏥 PulseQuery AI with Energy Metrics Loading...');
                    updateSystemStatus();
                    checkHealth();
                    
                    // Add Enter key support for password field
                    document.getElementById('password').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            login();
                        }
                    });
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
                    document.getElementById('systemResults').innerHTML = '<div class="alert alert-info">🔄 Performing health check...</div>';
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
                        if (data.prompt_optimizer) {
                            html += '<p><strong>Prompt Optimizer:</strong> ' + data.prompt_optimizer + '</p>';
                        }
                        html += '</div>';
                        document.getElementById('systemResults').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('systemResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ RAG Check Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function checkOptimizerStatus() {
                    if (!sessionId) {
                        alert('Please login first to check optimizer status');
                        return;
                    }
                    
                    document.getElementById('systemResults').innerHTML = '<div class="alert alert-info">🔄 Checking prompt optimizer status...</div>';
                    fetch('/api/debug/optimizer-status', {
                        headers: { 'X-Session-ID': sessionId }
                    })
                    .then(response => response.json())
                    .then(data => {
                        let html = '<div class="alert alert-warning"><h6>🧠 Prompt Optimizer Debug Status</h6>';
                        html += '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        html += '</div>';
                        document.getElementById('systemResults').innerHTML = html;
                    })
                    .catch(error => {
                        document.getElementById('systemResults').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ Optimizer Debug Failed</h6><p>' + error + '</p></div>';
                    });
                }

                function clearResults() {
                    document.getElementById('systemResults').innerHTML = '';
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
                            
                            // Show prompt optimization section
                            document.getElementById('promptOptimizationCard').style.display = 'block';
                            
                            // Show upload card if user has write permissions
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
                        document.getElementById('promptOptimizationCard').style.display = 'none';
                        document.getElementById('uploadCard').style.display = 'none';
                        resetOptimization();
                        alert('✅ Logged out successfully');
                    });
                }

                // ✅ Enhanced Prompt Optimization with Energy Metrics
                function optimizePrompt() {
                    const query = document.getElementById('userQuery').value.trim();
                    const useContext = document.getElementById('useContext').checked;
                    
                    if (!query) {
                        alert('Please enter a medical query first');
                        return;
                    }
                    
                    if (!sessionId) {
                        alert('Please login first');
                        return;
                    }
                    
                    // Show loading
                    document.getElementById('optimizedPromptSection').style.display = 'none';
                    document.getElementById('aiResponseSection').style.display = 'none';
                    document.getElementById('energyMetricsSection').style.display = 'none';
                    document.getElementById('promptMetrics').style.display = 'none';
                    
                    const loadingDiv = document.createElement('div');
                    loadingDiv.id = 'optimizationLoading';
                    loadingDiv.className = 'alert alert-info';
                    loadingDiv.innerHTML = '<div class="loading-spinner"></div>🔄 Analyzing query, optimizing prompt, and calculating energy savings...';
                    document.getElementById('promptOptimizationCard').appendChild(loadingDiv);
                    
                    fetch('/api/prompt/optimize', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Session-ID': sessionId
                        },
                        body: JSON.stringify({
                            query: query,
                            use_context: useContext
                        })
                    })
                    .then(response => {
                        console.log('📊 Optimization response status:', response.status);
                        if (!response.ok) {
                            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                        }
                        return response.json();
                    })
                    .then(data => {
                        console.log('📊 Optimization response data:', data);
                        
                        // Remove loading
                        const loading = document.getElementById('optimizationLoading');
                        if (loading) loading.remove();
                        
                        if (data.success) {
                            // Show optimized prompt
                            document.getElementById('optimizedPrompt').value = data.optimized_prompt;
                            
                            // Show optimization info
                            const patientName = data.patient_info.name || 'Not specified';
                            const patientAge = data.patient_info.age ? data.patient_info.age + ' years old' : 'Age not specified';
                            const chiefComplaint = data.patient_info.chief_complaint || 'Not specified';
                            
                            document.getElementById('optimizationInfo').innerHTML = 
                                `<strong>🔍 Query Analysis Results:</strong><br>` +
                                `<strong>Query Type:</strong> ${data.query_type} | ` +
                                `<strong>Medical Specialty:</strong> ${data.medical_specialty} | ` +
                                `<strong>Context Docs Used:</strong> ${data.context_docs_used}<br>` +
                                `<strong>Patient:</strong> ${patientName} (${patientAge}) | ` +
                                `<strong>Chief Complaint:</strong> ${chiefComplaint}`;
                            
                            // ✅ NEW: Display prompt metrics in card format
                            if (data.metrics) {
                                displayPromptMetrics(data.metrics);
                            }
                            
                            // ✅ NEW: Display Energy Metrics
                            if (data.energy_metrics) {
                                displayEnergyMetrics(data.energy_metrics);
                            }
                            
                            document.getElementById('optimizedPromptSection').style.display = 'block';
                            document.getElementById('optimizedPromptSection').scrollIntoView({ behavior: 'smooth' });
                            
                            // Show fallback warning if applicable
                            if (data.fallback_used) {
                                alert('⚠️ Note: Optimization used fallback mode due to: ' + (data.optimization_error || 'Unknown error'));
                            }
                            
                        } else {
                            alert('❌ Optimization failed: ' + (data.error || 'Unknown error'));
                        }
                    })
                    .catch(error => {
                        console.error('❌ Optimization error:', error);
                        const loading = document.getElementById('optimizationLoading');
                        if (loading) loading.remove();
                        alert('❌ Optimization error: ' + error.message);
                    });
                }

                // ✅ NEW: Display Prompt Metrics Function
                function displayPromptMetrics(metricsData) {
                    const promptSection = document.getElementById('promptMetrics');
                    const metricsGrid = document.getElementById('promptMetricsGrid');
                    
                    // Display prompt metric cards
                    metricsGrid.innerHTML = `
                        <div class="col-md-6">
                            <div class="prompt-metric">
                                <div class="prompt-metric-value">${metricsData.length}</div>
                                <div class="prompt-metric-label">Characters</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="prompt-metric">
                                <div class="prompt-metric-value">~${metricsData.token_estimate}</div>
                                <div class="prompt-metric-label">Est. Tokens</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="prompt-metric">
                                <div class="prompt-metric-value">${(metricsData.patient_specificity * 100).toFixed(0)}%</div>
                                <div class="prompt-metric-label">Patient Info</div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="prompt-metric">
                                <div class="prompt-metric-value">${(metricsData.medical_terminology_density * 100).toFixed(0)}%</div>
                                <div class="prompt-metric-label">Medical Terms</div>
                            </div>
                        </div>
                        <div class="col-md-12">
                            <div class="prompt-metric">
                                <div class="prompt-metric-value">${(metricsData.context_utilization * 100).toFixed(0)}%</div>
                                <div class="prompt-metric-label">Context Utilization</div>
                            </div>
                        </div>
                    `;
                    
                    promptSection.style.display = 'block';
                }

                // ✅ NEW: Display Energy Metrics Function
                function displayEnergyMetrics(energyData) {
                    const energySection = document.getElementById('energyMetricsSection');
                    const energyGrid = document.getElementById('energyMetricsGrid');
                    const environmentalImpact = document.getElementById('environmentalImpact');
                    
                    // Display energy metric cards
                    energyGrid.innerHTML = `
                        <div class="col-md-3">
                            <div class="energy-metric">
                                <div class="energy-value">${energyData.tokens_reduced}</div>
                                <div class="energy-label">Tokens Saved</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="energy-metric">
                                <div class="energy-value">${energyData.energy_saved_wh.toFixed(6)} Wh</div>
                                <div class="energy-label">Energy Saved</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="energy-metric">
                                <div class="energy-value">${energyData.efficiency_improvement_percent}%</div>
                                <div class="energy-label">Efficiency Gain</div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="energy-metric">
                                <div class="energy-value">$${energyData.cost_saved_usd.toFixed(6)}</div>
                                <div class="energy-label">Cost Savings</div>
                            </div>
                        </div>
                    `;
                    
                    // Environmental impact summary
                    environmentalImpact.innerHTML = `
                        <p><strong>🌱 Environmental Impact:</strong> 
                           ${energyData.co2_saved_kg.toFixed(8)} kg CO₂ saved per optimization
                        </p>
                        <p><strong>⚡ Energy Efficiency:</strong> 
                           ${energyData.energy_efficiency_percent}% improvement in energy utilization
                        </p>
                    `;
                    
                    // Create energy chart
                    createEnergyChart(energyData);
                    
                    energySection.style.display = 'block';
                }

                // ✅ NEW: Create Energy Chart
                function createEnergyChart(energyData) {
                    const ctx = document.getElementById('energyChart').getContext('2d');
                    
                    // Destroy existing chart if it exists
                    if (window.energyChartInstance) {
                        window.energyChartInstance.destroy();
                    }
                    
                    window.energyChartInstance = new Chart(ctx, {
                        type: 'bar',
                        data: {
                            labels: ['Original Tokens', 'Optimized Tokens', 'Tokens Saved'],
                            datasets: [{
                                data: [
                                    energyData.original_tokens,
                                    energyData.optimized_tokens,
                                    energyData.tokens_reduced
                                ],
                                backgroundColor: ['#ff6b6b', '#4ecdc4', '#45b7d1'],
                                borderColor: ['#ff5252', '#26a69a', '#2196f3'],
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            plugins: {
                                legend: {
                                    display: false
                                },
                                title: {
                                    display: true,
                                    text: 'Token Optimization Impact'
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    title: {
                                        display: true,
                                        text: 'Token Count'
                                    }
                                }
                            }
                        }
                    });
                }

                // ✅ ENHANCED: Generate with MARKDOWN RENDERING
                function generateFromPrompt() {
                    const finalPrompt = document.getElementById('optimizedPrompt').value.trim();
                    
                    if (!finalPrompt) {
                        alert('Please provide a prompt for generation');
                        return;
                    }
                    
                    if (!sessionId) {
                        alert('Please login first');
                        return;
                    }
                    
                    // Show loading state
                    document.getElementById('aiResponseSection').style.display = 'none';
                    
                    const loadingDiv = document.createElement('div');
                    loadingDiv.id = 'generationLoading';
                    loadingDiv.className = 'alert alert-warning';
                    loadingDiv.innerHTML = '<div class="loading-spinner"></div>🤖 Generating AI response from optimized prompt...';
                    
                    const promptCard = document.getElementById('promptOptimizationCard');
                    promptCard.appendChild(loadingDiv);
                    loadingDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    
                    fetch('/api/prompt/generate-final', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-Session-ID': sessionId
                        },
                        body: JSON.stringify({
                            final_prompt: finalPrompt,
                            max_tokens: 600,
                            temperature: 0.3
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        const loading = document.getElementById('generationLoading');
                        if (loading) loading.remove();
                        
                        if (data.success) {
                            // ✅ RENDER MARKDOWN: Convert markdown to HTML
                            const markdownText = data.ai_response;
                            const htmlContent = marked.parse(markdownText);
                            const responseElement = document.getElementById('aiResponse');
                            
                            // Set the HTML content and store original markdown for copying
                            responseElement.innerHTML = htmlContent;
                            responseElement.setAttribute('data-original-markdown', markdownText);
                            
                            // Show generation info
                            document.getElementById('generationInfo').innerHTML = 
                                `<strong>📊 Generation Info:</strong> ` +
                                `Response: ${data.generation_info.response_length} chars | ` +
                                `Device: ${data.generation_info.device} | ` +
                                `Generated: ${new Date(data.timestamp).toLocaleTimeString()} | ` +
                                `Milestone: ${data.milestone} | ` +
                                `Format: Rendered Markdown ✅`;
                            
                            document.getElementById('aiResponseSection').style.display = 'block';
                            document.getElementById('aiResponseSection').scrollIntoView({ 
                                behavior: 'smooth', 
                                block: 'start' 
                            });
                            
                        } else {
                            alert('❌ Generation failed: ' + data.error);
                        }
                    })
                    .catch(error => {
                        const loading = document.getElementById('generationLoading');
                        if (loading) loading.remove();
                        alert('❌ Generation error: ' + error);
                    });
                }

                function resetOptimization() {
                    document.getElementById('userQuery').value = '';
                    document.getElementById('optimizedPromptSection').style.display = 'none';
                    document.getElementById('aiResponseSection').style.display = 'none';
                    document.getElementById('energyMetricsSection').style.display = 'none';
                    document.getElementById('promptMetrics').style.display = 'none';
                    
                    // Remove any loading indicators
                    const loadingElements = document.querySelectorAll('#optimizationLoading, #generationLoading');
                    loadingElements.forEach(el => el.remove());
                    
                    // Destroy existing energy chart
                    if (window.energyChartInstance) {
                        window.energyChartInstance.destroy();
                        window.energyChartInstance = null;
                    }
                    
                    // Scroll back to top of form
                    document.getElementById('userQuery').scrollIntoView({ 
                        behavior: 'smooth', 
                        block: 'center' 
                    });
                    document.getElementById('userQuery').focus();
                }

                function resetForNew() {
                    resetOptimization();
                }

                // ✅ ENHANCED: Copy original markdown text
                function copyResponse() {
                    const responseElement = document.getElementById('aiResponse');
                    const originalMarkdown = responseElement.getAttribute('data-original-markdown');
                    const textToCopy = originalMarkdown || responseElement.textContent;
                    
                    navigator.clipboard.writeText(textToCopy).then(function() {
                        // Show success message
                        document.getElementById('systemResults').innerHTML = 
                            '<div class="alert alert-success">✅ Response copied to clipboard (original markdown format)!</div>';
                        setTimeout(() => {
                            document.getElementById('systemResults').innerHTML = '';
                        }, 3000);
                    }).catch(function(err) {
                        alert('❌ Could not copy response: ' + err);
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
                        '<div class="alert alert-info"><div class="loading-spinner"></div>Processing document with medical embeddings...</div>';
                    document.getElementById('uploadResult').style.display = 'block';
                    
                    fetch('/api/rag/upload', {
                        method: 'POST',
                        headers: { 'X-Session-ID': sessionId },
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            document.getElementById('uploadResult').innerHTML = 
                                '<div class="alert alert-success">' +
                                '<h6>✅ Document Processed Successfully!</h6>' +
                                '<p><strong>File:</strong> ' + (data.original_filename || 'Unknown') + '</p>' +
                                '<p><strong>Chunks:</strong> ' + (data.chunks_created || 0) + '</p>' +
                                '<p><strong>Processing:</strong> ' + (data.processing_method || 'Standard') + '</p>' +
                                '<p><strong>Persistent Storage:</strong> ' + (data.persistent_storage ? 'Yes' : 'No') + '</p>' +
                                '</div>';
                            fileInput.value = '';
                        } else {
                            document.getElementById('uploadResult').innerHTML = 
                                '<div class="alert alert-danger"><h6>❌ Upload Failed</h6><p>' + (data.error || 'Unknown error') + '</p></div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('uploadResult').innerHTML = 
                            '<div class="alert alert-danger"><h6>❌ Upload Error</h6><p>' + error + '</p></div>';
                    });
                });
            </script>
        </body>
        </html>
        """

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found", "milestone": 5}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "milestone": 5}), 500

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
    logger.info("🚀 Starting PulseQuery AI - Milestone 5: Enhanced Prompt Optimization with Energy Metrics")
    logger.info("🔧 Features: English Medical Prompt Optimizer, Query Classification, Medical Specialization")
    logger.info("🧠 Enhanced: Template-based Medical Prompt Engineering with Energy Savings")
    logger.info("📄 NEW: Interactive Prompt Optimization Workflow with Markdown Rendering + Energy Analytics")
    logger.info("🌱 GREEN: Energy consumption tracking and environmental impact metrics")
    logger.info("🎯 UI: Complete English-focused Medical AI System with Sustainability Dashboard")
    
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
        
        logger.info("\n🧪 Milestone 5 Enhanced Endpoints:")
        logger.info("   - /api/prompt/optimize - English prompt optimization + energy metrics")
        logger.info("   - /api/prompt/generate-final - Generate from optimized prompt")
        logger.info("   - /api/debug/optimizer-status - Debug optimizer status")
        logger.info("   - /api/auth/login - User authentication")
        logger.info("   - /api/rag/upload - Document upload with persistent storage")
        logger.info("   - /api/rag/stats - RAG system statistics")
        logger.info("   - /api/rag/persistence-status - Check document persistence")
        
        logger.info("\n🎯 Milestone 5 Enhanced Features:")
        logger.info("   - English Medical Prompt Optimizer with Energy Analytics")
        logger.info("   - Query Type Classification (7 types)")
        logger.info("   - Medical Specialty Detection (10+ specialties)")
        logger.info("   - Patient Information Extraction")
        logger.info("   - Template-based Prompt Generation")
        logger.info("   - Quality Metrics Calculation")
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
        logger.info("   - 📊 Energy metrics dashboard with Chart.js")
        logger.info("   - Real-time CO₂ and cost savings display")
        logger.info("   - Interactive energy impact charts")
        logger.info("   - Headers, lists, tables, and formatting support")
        logger.info("   - Copy original markdown functionality")
        logger.info("   - Elevated logo frame with hover effects")
        logger.info("   - 📊 Prompt metrics matching energy metrics styling")
        
        logger.info("\n🔋 Energy Analytics Features:")
        logger.info("   - Token usage before/after optimization")
        logger.info("   - Energy consumption in Wh (Watt-hours)")
        logger.info("   - CO₂ emissions saved (kg)")
        logger.info("   - Cost savings in USD")
        logger.info("   - Efficiency improvement percentages")
        logger.info("   - Visual charts for impact analysis")
        logger.info("   - Consistent card-based metrics display")
        
        logger.info("\n🎯 UI Enhancement Details:")
        logger.info("   - Prompt metrics now match energy metrics styling")
        logger.info("   - Consistent gradient backgrounds and card layouts")
        logger.info("   - Professional medical dashboard appearance")
        logger.info("   - Interactive charts for both prompt quality and energy impact")
        logger.info("   - Unified color scheme across all metrics sections")
        
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
        logger.info("\n✅ Milestone 5 Enhanced shutdown complete")

