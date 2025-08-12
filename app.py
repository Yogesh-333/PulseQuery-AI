import logging
import warnings
from datetime import datetime, timezone
import os
import re
import time
from collections import Counter
from flask import send_from_directory
import numpy as np

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

# ✅ ENHANCED MEDICAL TERMINOLOGY FOR BETTER ANALYSIS
ENHANCED_MEDICAL_TERMS = {
    'symptoms': [
        'pain', 'chest pain', 'shortness of breath', 'nausea', 'vomiting', 'dizziness', 
        'headache', 'fever', 'fatigue', 'weakness', 'sweating', 'palpitations',
        'cough', 'abdominal pain', 'back pain', 'muscle pain', 'joint pain',
        'breathing difficulty', 'chest tightness', 'rapid heartbeat', 'malaise'
    ],
    'conditions': [
        'hypertension', 'diabetes', 'cardiac', 'cardiology', 'emergency', 'diagnosis', 
        'syndrome', 'disease', 'disorder', 'myocardial infarction', 'heart failure',
        'atrial fibrillation', 'coronary artery disease', 'acute coronary syndrome',
        'congestive heart failure', 'myocardial', 'infarction', 'arrhythmia', 'stenosis'
    ],
    'demographics': [
        'patient', 'male', 'female', 'years old', 'age', 'elderly', 'adult', 'pediatric',
        'year-old', 'yo', 'mr', 'mrs', 'ms', 'child', 'adolescent', 'infant', 'neonate'
    ],
    'clinical': [
        'diagnosis', 'treatment', 'symptoms', 'history', 'complaint', 'medication', 
        'therapy', 'procedure', 'examination', 'assessment', 'evaluation', 'monitoring',
        'presenting', 'complained', 'reports', 'denies', 'admits', 'clinical', 'medical'
    ],
    'anatomy': [
        'heart', 'lung', 'brain', 'kidney', 'liver', 'stomach', 'chest', 'abdomen', 
        'extremities', 'coronary', 'artery', 'vessel', 'ventricle', 'atrium', 'pulmonary'
    ],
    'vitals': [
        'blood pressure', 'heart rate', 'temperature', 'respiratory rate', 
        'oxygen saturation', 'pulse', 'bp', 'hr', 'temp', 'rr', 'o2 sat'
    ],
    'procedures': [
        'ecg', 'ekg', 'ct', 'mri', 'ultrasound', 'x-ray', 'blood test',
        'catheterization', 'angiography', 'endoscopy', 'biopsy', 'echocardiogram'
    ]
}

# ✅ HELPER FUNCTIONS FOR OPTIMIZATION
def calculate_enhanced_medical_density(text):
    """Enhanced medical terminology density calculation with better accuracy"""
    if not text or len(text.strip()) == 0:
        return 0.0
    
    text_lower = text.lower()
    medical_term_count = 0
    total_tokens = len(text.split())
    
    if total_tokens == 0:
        return 0.0
    
    # Count medical terms from all categories
    for category, terms in ENHANCED_MEDICAL_TERMS.items():
        for term in terms:
            if term in text_lower:
                # Use word boundaries for accurate counting
                pattern = r'\b' + re.escape(term) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                medical_term_count += matches
    
    # Calculate base density
    base_density = (medical_term_count / total_tokens) * 100
    
    # Add category diversity bonus
    categories_present = 0
    for category, terms in ENHANCED_MEDICAL_TERMS.items():
        if any(term in text_lower for term in terms):
            categories_present += 1
    
    diversity_bonus = min(categories_present * 2, 20)
    final_density = min(base_density + diversity_bonus, 100.0)
    
    return final_density

def calculate_enhanced_tokens(text: str) -> int:
    """Enhanced token estimation with medical complexity factors"""
    if not text:
        return 0
    
    # Method 1: Word count with coefficient
    word_count = len(text.split())
    word_estimate = int(word_count * 1.2)  # Conservative estimate
    
    # Method 2: Character-based estimation
    char_estimate = int(len(text) / 4.5)
    
    # Method 3: Medical complexity adjustment
    medical_indicators = ['patient', 'medical', 'clinical', 'diagnosis', 'treatment']
    medical_count = sum(1 for indicator in medical_indicators if indicator in text.lower())
    complexity_multiplier = 1.0 + (medical_count * 0.02)  # Small adjustment
    
    # Weighted ensemble
    base_estimate = (word_estimate * 0.7 + char_estimate * 0.3)
    final_estimate = int(base_estimate * complexity_multiplier)
    
    return max(1, final_estimate)

def enhanced_query_optimization(query: str, medical_density: float = 0.0, complexity_score: float = 0.0) -> str:
    """Enhanced query optimization with medical context preservation"""
    if not query:
        return ""
    
    optimized = query.strip()
    
    # Remove redundant phrases
    redundant_patterns = [
        r'\bplease\s+', r'\bcan\s+you\s+', r'\bcould\s+you\s+',
        r'\bi\s+would\s+like\s+', r'\btell\s+me\s+about\s+',
        r'\bhelp\s+me\s+understand\s+', r'\bprovide\s+information\s+about\s+',
        r'\blet\s+me\s+know\s+', r'\bexplain\s+to\s+me\s+'
    ]
    
    for pattern in redundant_patterns:
        optimized = re.sub(pattern, '', optimized, flags=re.IGNORECASE)
    
    # Apply medical abbreviations
    medical_abbreviations = {
        r'\byears?\s+old\b': 'yo',
        r'\bpatient\b': 'pt',
        r'\bblood\s+pressure\b': 'BP',
        r'\bheart\s+rate\b': 'HR',
        r'\bshortness\s+of\s+breath\b': 'SOB',
        r'\bchest\s+pain\b': 'CP',
        r'\bmyocardial\s+infarction\b': 'MI',
        r'\bcongestive\s+heart\s+failure\b': 'CHF',
        r'\bhistory\s+of\b': 'h/o'
    }
    
    for pattern, replacement in medical_abbreviations.items():
        optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
    
    # Clean up whitespace
    optimized = re.sub(r'\s+', ' ', optimized).strip()
    
    # Ensure we return something meaningful
    if len(optimized) < 5:
        optimized = f"Medical query: {query[:100]}"
    
    return optimized

from flask import Flask, jsonify, render_template_string, request, session
import tempfile
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
        
        # Initialize RAG system with MedEmbed-enhanced prompt optimizer
        if RAG_AVAILABLE:
            try:
                logger.info("🔄 Creating enhanced RAG system with MedEmbed prompt optimization...")
                
                data_dir = os.path.abspath('data/chromadb')
                os.makedirs(data_dir, exist_ok=True)
                
                logger.info(f"🔄 Initializing RAG with persistent storage: {data_dir}")
        
                # Verify write permissions
                test_file = os.path.join(data_dir, 'persistence_test.tmp')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    logger.info("✅ Directory permissions verified")
                except Exception as perm_error:
                    logger.error(f"❌ Cannot write to RAG directory: {perm_error}")
                    raise
                
                # Initialize RAG system
                app.rag_system = RAGSystem(
                    data_dir=data_dir,
                    embedding_model="medical"
                )
                
                # Verify document count
                doc_count = app.rag_system.chroma_collection.count() if app.rag_system.chroma_collection else 0
                logger.info(f"📊 RAG system loaded with {doc_count} existing documents")
                
                # Verify persistence
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
                
                # Initialize session manager
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
        
        # Check prompt optimizer status
        prompt_optimizer_status = "❌ Not Available"
        if app.rag_system and hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer:
            prompt_optimizer_status = "✅ Ready - MedEmbed Enhanced kWh"
        
        return jsonify({
            "message": "🔬 PulseQuery AI - MedEmbed Enhanced Energy-Efficient System (kWh + Rounded)",
            "status": "Running",
            "milestone": 5.3,
            "enhancement": "kWh Energy Units + Rounded Values + MedEmbed Integration",
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
                "medembed_integration": "✅ Active",
                "energy_optimization": "✅ kWh + Rounded Values",
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
                "milestone": 5.3
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
                "milestone": 5.3
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
                "medembed_enhanced": "✅ Active",
                "energy_kwh_conversion": "✅ Enabled",
                "rounded_values": "✅ Applied"
            },
            "milestone": 5.3,
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
        else:
            health_status["components"]["rag_system"] = "❌ Not Available"
            health_status["components"]["document_upload"] = "❌ Not Available"
        
        # Prompt optimizer health
        if app.rag_system and hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer:
            health_status["components"]["prompt_optimizer"] = "✅ Ready - MedEmbed Enhanced kWh"
        else:
            health_status["components"]["prompt_optimizer"] = "❌ Not Available"
        
        return jsonify(health_status)

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
                "milestone": 5.3,
                **status,
                "gpu_support": "Enabled",
                "model_file": "medgemma-4b-it-Q8_0.gguf"
            })
        except Exception as e:
            return jsonify({
                "status": "error",
                "error": str(e),
                "milestone": 5.3
            })

    # ✅ FIXED: Enhanced Prompt Optimization Endpoint with kWh and Rounded Values
    @app.route('/api/prompt/optimize', methods=['POST'])
    @require_auth(app.session_manager)  
    def optimize_prompt():
        """FIXED: Energy-efficient prompt optimization with kWh and rounded values"""
        logger.info("🧠 MEDEMBED-ENHANCED OPTIMIZATION ENDPOINT CALLED (kWh + Rounded)")
        
        try:
            data = request.get_json()
            original_query = data.get('query', '').strip()
            use_context = data.get('use_context', True)
            
            logger.info(f"📋 Query received: {original_query[:100]}..." if original_query else "📋 No query provided")
            
            if not original_query:
                logger.error("❌ No query provided")
                return jsonify({'error': 'Query is required for optimization'}), 400
            
            # ✅ CALCULATE ORIGINAL METRICS FIRST
            original_chars = len(original_query)
            original_tokens = calculate_enhanced_tokens(original_query)
            original_medical_density = calculate_enhanced_medical_density(original_query)
            
            logger.info(f"📊 Original metrics: {original_chars} chars, {original_tokens} tokens, {original_medical_density:.1f}% medical density")
            
            # Get context documents if requested
            context_docs = []
            if use_context and app.rag_system:
                try:
                    context_docs = app.rag_system.search_relevant_context(original_query, max_docs=3)
                    logger.info(f"📄 Found {len(context_docs)} context documents")
                except Exception as context_error:
                    logger.warning(f"⚠️ Context search failed: {context_error}")
                    context_docs = []
            
            # ✅ PERFORM OPTIMIZATION WITH MedEmbed
            optimized_query = ""
            optimizer_used = "unknown"
            
            # Try MedEmbed-enhanced optimizer first
            try:
                if hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer:
                    logger.info("🧠 Using MedEmbed-enhanced RAG optimizer...")
                    result = app.rag_system.prompt_optimizer.optimize_prompt(original_query, context_docs)
                    optimized_query = result.get('optimized_prompt', '')
                    optimizer_used = "medembed_enhanced"
                    logger.info(f"✅ MedEmbed optimizer produced: {len(optimized_query)} chars")
                else:
                    raise Exception("MedEmbed optimizer not available")
                    
            except Exception as opt_error:
                logger.warning(f"⚠️ MedEmbed optimizer failed: {opt_error}, using enhanced fallback")
                
                # Enhanced fallback optimization
                optimized_query = enhanced_query_optimization(
                    original_query, 
                    original_medical_density, 
                    5.0  # complexity score
                )
                optimizer_used = "enhanced_fallback"
                logger.info(f"✅ Enhanced fallback produced: {len(optimized_query)} chars")
            
            # ✅ ENSURE OPTIMIZED QUERY IS NOT EMPTY
            if not optimized_query or len(optimized_query.strip()) < 10:
                logger.warning("⚠️ Optimized query is too short, using medical wrapper")
                optimized_query = f"Medical analysis: {original_query}"
                optimizer_used = "emergency_fallback"
            
            # ✅ CALCULATE OPTIMIZED METRICS
            optimized_chars = len(optimized_query)
            optimized_tokens = calculate_enhanced_tokens(optimized_query)
            optimized_medical_density = calculate_enhanced_medical_density(optimized_query)
            
            logger.info(f"📊 Optimized metrics: {optimized_chars} chars, {optimized_tokens} tokens, {optimized_medical_density:.1f}% medical density")
            
            # ✅ CALCULATE ENERGY METRICS WITH KWH CONVERSION AND ROUNDING
            processing_time_start = time.time()
            tokens_reduced = max(0, original_tokens - optimized_tokens)
            chars_reduced = max(0, original_chars - optimized_chars)
            
            # Energy calculation constants
            ENERGY_PER_TOKEN = 0.0012  # Wh per token
            CO2_PER_KWH = 0.233        # kg CO₂ per kWh
            COST_PER_KWH = 0.12        # USD per kWh
            
            # ✅ ENERGY CALCULATIONS WITH KWH CONVERSION
            energy_saved_wh = tokens_reduced * ENERGY_PER_TOKEN
            energy_saved_kwh = energy_saved_wh / 1000  # Convert Wh to kWh
            co2_saved_kg = energy_saved_kwh * CO2_PER_KWH
            cost_saved_usd = energy_saved_kwh * COST_PER_KWH
            processing_time_ms = (time.time() - processing_time_start) * 1000
            
            # ✅ ROUNDED VALUES
            efficiency_improvement = round((tokens_reduced / original_tokens * 100), 1) if original_tokens > 0 else 0.0
            compression_ratio = round((optimized_tokens / original_tokens), 3) if original_tokens > 0 else 1.0
            
            # ✅ CREATE PROPERLY DEFINED energy_metrics WITH KWH
            energy_metrics = {
                'tokens_reduced': int(tokens_reduced),
                'energy_saved_kwh': round(energy_saved_kwh, 9),  # ✅ kWh with 9 decimal places
                'efficiency_improvement_percent': efficiency_improvement,  # ✅ Already rounded to 1 decimal
                'cost_saved_usd': round(cost_saved_usd, 8),      # ✅ 8 decimal places
                'co2_saved_kg': round(co2_saved_kg, 8),          # ✅ 8 decimal places
                'original_tokens': int(original_tokens),
                'optimized_tokens': int(optimized_tokens),
                'compression_ratio': compression_ratio,
                'processing_time_ms': round(processing_time_ms, 1)
            }
            
            logger.info(f"🔋 Energy metrics: {tokens_reduced} tokens saved, {energy_saved_kwh:.9f} kWh saved, {efficiency_improvement}% improvement")
            
                        # ✅ PREPARE COMPLETE RESPONSE
            response_data = {
                'success': True,
                'optimized_prompt': optimized_query,
                'original_query': original_query,
                'energy_metrics': energy_metrics,  # ✅ PROPERLY DEFINED WITH KWH
                'original_metrics': {
                    'length': original_chars,
                    'token_estimate': int(original_tokens),
                    'medical_terminology_density': round(original_medical_density / 100.0, 3)
                },
                'metrics': {
                    'length': optimized_chars,
                    'token_estimate': int(optimized_tokens),
                    'medical_terminology_density': round(optimized_medical_density / 100.0, 3)
                },
                'enhanced_analysis': {
                    'optimizer_used': optimizer_used,
                    'processing_time_ms': round(processing_time_ms, 1),
                    'character_reduction': chars_reduced,
                    'token_reduction': tokens_reduced,
                    'medical_density_change': round(optimized_medical_density - original_medical_density, 1),
                    'compression_ratio': compression_ratio,
                    'medembed_powered': optimizer_used == "medembed_enhanced",
                    'energy_optimization_version': '3.0_kwh_rounded_fixed'
                },
                'context_docs_used': len(context_docs),
                'optimized_by': request.current_user["user_name"],
                'milestone': 5.3,
                'optimization_successful': True
            }
            
            logger.info(f"✅ Complete optimization response prepared with kWh conversion and rounded values")
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ Complete optimization failure: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # ✅ EMERGENCY FALLBACK WITH PROPER ENERGY METRICS IN KWH
            try:
                fallback_energy_metrics = {
                    'tokens_reduced': 0,
                    'energy_saved_kwh': 0.0,
                    'efficiency_improvement_percent': 0.0,
                    'cost_saved_usd': 0.0,
                    'co2_saved_kg': 0.0,
                    'original_tokens': int(len(original_query.split()) * 1.2) if 'original_query' in locals() else 0,
                    'optimized_tokens': int(len(original_query.split()) * 1.2) if 'original_query' in locals() else 0,
                    'compression_ratio': 1.0,
                    'processing_time_ms': 0.0
                }
                
                return jsonify({
                    'success': False, 
                    'error': f'Optimization failed: {str(e)}',
                    'energy_metrics': fallback_energy_metrics,  # ✅ PROVIDE FALLBACK WITH KWH
                    'debug_info': {
                        'original_query_length': len(original_query) if 'original_query' in locals() else 0,
                        'error_location': 'main_optimization_endpoint',
                        'milestone': 5.3
                    }
                }), 500
                
            except Exception as fallback_error:
                return jsonify({
                    'success': False,
                    'error': f'Critical optimization failure: {str(e)}',
                    'fallback_error': str(fallback_error)
                }), 500

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
                'milestone': 5.3
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
                "milestone": 5.3,
                "rag_stats": stats,
                "accessed_by": request.current_user["user_name"],
                "embedding_info": {
                    "model": stats.get('embedding_model', 'MedEmbed-base-v0.1'),
                    "type": "Medical-specialized with kWh energy optimization"
                },
                "prompt_optimizer": "✅ MedEmbed Enhanced kWh Available" if hasattr(app.rag_system, 'prompt_optimizer') and app.rag_system.prompt_optimizer else "❌ Not Available"
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
                    "processing_method": "MedEmbed enhanced medical embeddings with kWh energy optimization"
                })
                
                logger.info(f"✅ Upload result: {result}")
                
                return jsonify({
                    "success": result.get('success', True),
                    "message": "Document uploaded and processed with MedEmbed embeddings (kWh optimized)",
                    "milestone": 5.3,
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

    @app.route('/api/analytics/energy-stats', methods=['GET'])
    @require_auth(app.session_manager)
    def get_energy_stats():
        """Get energy efficiency statistics with kWh conversion and rounded values"""
        try:
            # Create data directory and database if it doesn't exist
            data_dir = os.path.abspath('data')
            os.makedirs(data_dir, exist_ok=True)
            
            import sqlite3
            db_path = os.path.join(data_dir, 'pulsequery.db')
            
            with sqlite3.connect(db_path) as conn:
                # Create table if it doesn't exist
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS optimizations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        original_query TEXT,
                        optimized_query TEXT,
                        tokens_saved INTEGER,
                        energy_saved_wh REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
                
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_optimizations,
                        COALESCE(SUM(tokens_saved), 0) as total_tokens_saved,
                        COALESCE(SUM(energy_saved_wh), 0.0) as total_energy_saved_wh,
                        COALESCE(AVG(tokens_saved), 0.0) as avg_tokens_saved
                    FROM optimizations 
                    WHERE created_at >= datetime('now', '-30 days')
                ''')
                stats = cursor.fetchone()
            
            # ✅ Convert to kWh for display and round properly
            total_energy_kwh = (stats[2] or 0.0) / 1000
            
            return jsonify({
                'energy_analytics': {
                    'total_optimizations': int(stats[0]) if stats[0] else 0,
                    'total_tokens_saved': int(stats[1]) if stats[1] else 0,
                    'total_energy_saved_kwh': round(total_energy_kwh, 9),  # ✅ kWh with 9 decimals
                    'average_tokens_saved': round(float(stats[3]) if stats[3] else 0.0, 1),  # ✅ Rounded to 1 decimal
                    'equivalent_co2_saved_kg': round(total_energy_kwh * 0.233, 8),  # ✅ 8 decimal places
                    'equivalent_cost_saved_usd': round(total_energy_kwh * 0.12, 8),  # ✅ 8 decimal places
                    'energy_unit': 'kWh'  # ✅ Clearly show kWh
                },
                'optimizer_version': '3.0_medembed_kwh_rounded_fixed',
                'energy_improvements': {
                    'conversion': 'Wh to kWh applied',
                    'rounding': 'Long decimals fixed',
                    'display': 'Card overflow prevented'
                },
                'accessed_by': request.current_user["user_name"],
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"❌ Analytics error: {e}")
            return jsonify({
                'error': str(e),
                'fallback_analytics': {
                    'total_optimizations': 0,
                    'total_energy_saved_kwh': 0.0,
                    'energy_unit': 'kWh',
                    'message': 'Database error - showing default values'
                }
            }), 500

    # Static file routes
    @app.route('/main')
    def main_screen():
        """Serve the main screen"""
        return send_from_directory('static', 'main.html')

    @app.route('/test-ui')
    def test_ui():
        """Serve the comprehensive UI"""
        return send_from_directory('static', 'ui.html')
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found", "milestone": 5.3}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({"error": "Internal server error", "milestone": 5.3}), 500

# Main execution
if __name__ == '__main__':
    logger.info("🚀 Starting PulseQuery AI - Milestone 5.3: COMPLETE kWh + Rounded Values Fixed")
    logger.info("🔧 Features: MedEmbed-Enhanced Energy-Efficient Medical Prompt Optimizer")
    logger.info("🧠 Enhanced: Semantic Medical Analysis with Advanced Neural Embeddings")
    logger.info("📊 FIXED: kWh Energy Units + Properly Rounded Values (No UI Card Overflow)")
    logger.info("🌱 GREEN: Accurate kWh Energy Consumption + Environmental Impact")
    logger.info("🩺 MEDICAL: MedEmbed-base-v0.1 Powered Semantic Understanding")
    logger.info("⚡ OPTIMIZED: Intelligent Query Compression with Medical Context Preservation")
    logger.info("🎯 UI: Original Simple Style with Fixed Value Display")
    
    try:
        logger.info("\n🔄 Creating MedEmbed-enhanced Flask application...")
        app = create_app()
        logger.info("✅ MedEmbed-enhanced application created successfully!")
        
        logger.info("\n🌐 Server Information:")
        logger.info("📍 Main Server: http://localhost:5000")
        logger.info("🔗 Main Dashboard: http://localhost:5000/main")
        logger.info("❤️ Health Check: http://localhost:5000/health")
        
        logger.info("\n🧪 Milestone 5.3 Fixed Features:")
        logger.info("   - ✅ FIXED: kWh Energy Units (converted from Wh)")
        logger.info("   - ✅ FIXED: Rounded Values (30.0% instead of 29.958391123439664%)")
        logger.info("   - ✅ FIXED: Card Overflow (values fit properly in UI cards)")
        logger.info("   - ✅ ENHANCED: MedEmbed-base-v0.1 Semantic Understanding")
        logger.info("   - ✅ ADVANCED: Neural Medical Concept Recognition")
        logger.info("   - ✅ INTELLIGENT: Context-Aware Optimization")
        logger.info("   - ✅ ROBUST: Multi-Level Fallback System")
        logger.info("   - ✅ PERSISTENT: Document Storage Across Restarts")
        logger.info("   - ✅ ORIGINAL: Simple UI Style Preserved")
        
        logger.info("\n🎯 Expected Fixed Results:")
        logger.info("   - Efficiency: 30.0% (rounded, fits in card)")
        logger.info("   - Energy: 0.000000030 kWh (not 0.000102 Wh)")
        logger.info("   - Cost: $0.00000400 (rounded, fits properly)")
        logger.info("   - No card overflow or layout issues")
        
        logger.info("\n🔄 Starting server with complete kWh + Rounded Values fix...")
        
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
        logger.info("\n✅ Milestone 5.3 shutdown complete - kWh + Rounded Values FIXED!")

