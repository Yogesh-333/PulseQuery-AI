"""
Energy-Efficient Medical Prompt Optimizer with MedEmbed Integration
Combines medical intelligence with aggressive energy optimization using semantic embeddings
"""

import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Optional ML / DL imports (graceful fallback)
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

import math
import random
from collections import Counter

# ===============================================================================
# ✅ ENERGY EFFICIENCY DATACLASSES & ENUMS
# ===============================================================================

@dataclass
class EnergyMetrics:
    """Energy efficiency metrics for optimization"""
    original_tokens: int
    optimized_tokens: int
    tokens_reduced: int
    energy_saved_kwh: float  # ✅ CHANGED: Now in kWh instead of Wh
    cost_saved_usd: float
    co2_reduced_kg: float
    efficiency_improvement_percent: float
    processing_time_ms: float
    compression_ratio: float

@dataclass
class OptimizationResult:
    """Complete optimization result with energy metrics"""
    optimized_prompt: str
    original_prompt: str
    energy_metrics: EnergyMetrics
    query_type: str
    medical_specialty: str
    patient_info: 'PatientInfo'
    metrics: 'PromptMetrics'
    language: str
    timestamp: str
    compression_techniques_used: List[str]

class QueryType(Enum):
    PATIENT_REPORT = "patient_report"
    SYMPTOM_ANALYSIS = "symptom_analysis" 
    TREATMENT_PLAN = "treatment_plan"
    DIAGNOSTIC_WORKUP = "diagnostic_workup"
    MEDICATION_REVIEW = "medication_review"
    FOLLOW_UP = "follow_up"
    GENERAL_MEDICAL = "general_medical"

class MedicalSpecialty(Enum):
    CARDIOLOGY = "cardiology"
    ENDOCRINOLOGY = "endocrinology"
    PULMONOLOGY = "pulmonology"
    NEUROLOGY = "neurology"
    GASTROENTEROLOGY = "gastroenterology"
    ORTHOPEDICS = "orthopedics"
    ONCOLOGY = "oncology"
    PSYCHIATRY = "psychiatry"
    DERMATOLOGY = "dermatology"
    GENERAL_MEDICINE = "general_medicine"

@dataclass
class PatientInfo:
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    mrn: Optional[str] = None
    chief_complaint: Optional[str] = None

@dataclass
class PromptMetrics:
    length: int
    token_estimate: int
    context_utilization: float
    patient_specificity: float
    medical_terminology_density: float

# ===============================================================================
# ✅ MEDEMBED-ENHANCED MEDICAL PROMPT OPTIMIZER
# ===============================================================================

class EnergyEfficientMedicalOptimizer:
    """
    Medical prompt optimizer using MedEmbed semantic understanding with energy efficiency
    """
    
    def __init__(self, max_prompt_length: int = 2500, max_context_docs: int = 3, 
                 energy_mode: bool = True, use_medembed: bool = True):
        self.max_prompt_length = max_prompt_length
        self.max_context_docs = max_context_docs
        self.energy_mode = energy_mode
        self.prompt_cache = {}
        
        # ✅ UPDATED: Energy calculation constants for kWh
        self.ENERGY_PER_TOKEN = 0.0012  # Wh per token (MedGemma 4B Q8) - will convert to kWh
        self.CO2_PER_KWH = 0.233        # kg CO₂ per kWh (updated for kWh)
        self.COST_PER_KWH = 0.12        # USD per kWh
        self.PROCESSING_OVERHEAD = 0.0001
        
        # Medical specialty keywords (condensed for efficiency)
        self.specialty_keywords = {
            MedicalSpecialty.CARDIOLOGY: [
                'heart', 'cardiac', 'chest pain', 'MI', 'CHF', 'arrhythmia', 'BP', 'ECG'
            ],
            MedicalSpecialty.ENDOCRINOLOGY: [
                'diabetes', 'thyroid', 'hormone', 'glucose', 'insulin', 'DM', 'TSH'
            ],
            MedicalSpecialty.PULMONOLOGY: [
                'lung', 'respiratory', 'SOB', 'cough', 'asthma', 'COPD', 'pneumonia'
            ],
            MedicalSpecialty.NEUROLOGY: [
                'brain', 'neurological', 'seizure', 'stroke', 'headache', 'weakness'
            ],
            MedicalSpecialty.GASTROENTEROLOGY: [
                'stomach', 'GI', 'abdominal', 'liver', 'nausea', 'diarrhea'
            ]
        }
        
        # Medical abbreviation dictionary for compression
        self.medical_compressions = {
            r'\byears?\s+old\b': 'yo',
            r'\byear\s*-\s*old\b': 'yo',
            r'\bold\s+male\b': 'M',
            r'\bold\s+female\b': 'F',
            r'\bmale\s+patient\b': 'M pt',
            r'\bfemale\s+patient\b': 'F pt',
            r'\bpatient\b': 'pt',
            r'\bblood\s+pressure\b': 'BP',
            r'\bheart\s+rate\b': 'HR',
            r'\bshortness\s+of\s+breath\b': 'SOB',
            r'\bchest\s+pain\b': 'CP',
            r'\babdominal\s+pain\b': 'abd pain',
            r'\bmyocardial\s+infarction\b': 'MI',
            r'\bcongestive\s+heart\s+failure\b': 'CHF',
            r'\batrial\s+fibrillation\b': 'AFib',
            r'\bcoronary\s+artery\s+disease\b': 'CAD',
            r'\bhypertension\b': 'HTN',
            r'\bdiabetes\s+mellitus\b': 'DM',
            r'\bhistory\s+of\b': 'h/o',
            r'\bfor\s+the\s+past\b': 'x',
            r'\bhours?\b': 'hrs',
            r'\bminutes?\b': 'mins',
            r'\btemperature\b': 'temp',
            r'\brespiratory\s+rate\b': 'RR',
            r'\boxygen\s+saturation\b': 'O2 sat'
        }
        
        # Initialize compact templates
        self._init_energy_efficient_templates()
        
        # ✅ MEDEMBED INTEGRATION: Use existing torch setup
        self.use_medembed = use_medembed and TORCH_AVAILABLE
        self._embedder_ready = False
        
        if TORCH_AVAILABLE:
            # Optimize device detection
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
                print("🚀 Using GPU for MedEmbed")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = torch.device("mps")
                print("🍎 Using MPS for MedEmbed")
            else:
                self._device = torch.device("cpu")
                print("🖥️ Using CPU for MedEmbed")
        
        if self.use_medembed:
            try:
                print(f"🔄 Loading MedEmbed-base-v0.1 on {self._device}...")
                
                self._tokenizer = AutoTokenizer.from_pretrained(
                    "abhinand/MedEmbed-base-v0.1", 
                    use_fast=True,
                    trust_remote_code=True
                )
                self._embedder = AutoModel.from_pretrained(
                    "abhinand/MedEmbed-base-v0.1",
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                
                self._embedder.to(self._device)
                self._embedder.eval()
                self._embedder_ready = True
                
                print(f"✅ MedEmbed ready on {self._device} with torch optimizations")
                
            except Exception as e:
                print(f"❌ MedEmbed loading failed: {e}")
                self._embedder_ready = False
        
        print(f"✅ Energy-Efficient Medical Optimizer initialized (MedEmbed: {'Ready' if self._embedder_ready else 'Fallback'})")
    
    def _init_energy_efficient_templates(self):
        """Compact templates for energy efficiency"""
        self.templates = {
            QueryType.PATIENT_REPORT: """{patient_name}, {patient_age}yo {patient_gender}, MRN:{mrn}
Chief complaint: {chief_complaint}
Context: {context_summary}

Medical assessment and treatment plan:""",

            QueryType.SYMPTOM_ANALYSIS: """{patient_name} w/ {symptom_description}
Context: {context_summary}

Differential diagnosis and workup:""",

            QueryType.TREATMENT_PLAN: """Treatment for: {condition_description}
Context: {context_summary}

Management plan:""",

            QueryType.GENERAL_MEDICAL: """Medical query: {query_text}
Context: {context_summary}

Clinical response:"""
        }
    
    # ===============================================================================
    # ✅ MEDEMBED SEMANTIC ANALYSIS METHODS
    # ===============================================================================
    
    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the loaded MedEmbed model"""
        if not self._embedder_ready:
            return self._fallback_features(texts)
        
        with torch.no_grad():
            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            if self._device:
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            outputs = self._embedder(**inputs)
            
            # Mean pooling
            if hasattr(outputs, "last_hidden_state"):
                hidden = outputs.last_hidden_state
                attention_mask = inputs.get("attention_mask")
                
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
                    sum_embeddings = torch.sum(hidden * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
                else:
                    embeddings = hidden.mean(dim=1)
            else:
                embeddings = outputs[0].mean(dim=1)
            
            # Convert to Python lists
            embeddings = embeddings.cpu().numpy()
            return [list(vec) for vec in embeddings]
    
    def _fallback_features(self, texts: List[str]) -> List[List[float]]:
        """Fallback feature extraction when MedEmbed is not available"""
        features = []
        for text in texts:
            words = text.lower().split()
            feature_vector = [
                len([w for w in words if 'patient' in w]),
                len([w for w in words if 'pain' in w]),
                len([w for w in words if 'heart' in w]),
                len([w for w in words if 'medical' in w]),
                len(words)  # Total word count
            ]
            features.append(feature_vector)
        return features
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def analyze_query_with_medembed(self, query: str, context_docs: List[Dict] = None) -> Dict[str, Any]:
        """Use MedEmbed embeddings for advanced query analysis"""
        if not self._embedder_ready:
            return self._fallback_analysis(query)
        
        print("🧠 Using MedEmbed for semantic query analysis...")
        
        # Generate embedding for the query
        query_embedding = self._embed_text([query])[0]
        
        # Define medical concept embeddings
        medical_concepts = {
            'emergency': "acute urgent emergency critical severe immediate life-threatening",
            'chronic': "chronic long-term ongoing persistent history management",
            'diagnostic': "diagnosis workup evaluation assessment investigation testing",
            'therapeutic': "treatment therapy medication management intervention plan",
            'symptoms': "pain discomfort symptoms signs presenting complaints",
            'cardiology': "heart cardiac chest pain blood pressure cardiovascular",
            'pulmonary': "lung respiratory breathing shortness breath cough",
            'endocrine': "diabetes thyroid hormone insulin glucose endocrine"
        }
        
        # Calculate semantic similarities
        concept_embeddings = self._embed_text(list(medical_concepts.values()))
        concept_scores = {}
        
        for i, (concept, description) in enumerate(medical_concepts.items()):
            similarity = self._cosine_similarity(query_embedding, concept_embeddings[i])
            concept_scores[concept] = similarity
        
        # Find dominant medical concepts
        top_concepts = sorted(concept_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"🎯 Top medical concepts: {[(c, f'{s:.3f}') for c, s in top_concepts]}")
        
        return {
            'concept_scores': concept_scores,
            'primary_concept': top_concepts[0][0] if top_concepts else 'general',
            'semantic_complexity': sum(concept_scores.values()) / len(concept_scores),
            'embedding_dimension': len(query_embedding)
        }
    
    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when MedEmbed is not available"""
        words = query.lower().split()
        medical_words = ['patient', 'pain', 'heart', 'medical', 'diagnosis', 'treatment']
        medical_count = sum(1 for word in words if any(med in word for med in medical_words))
        
        return {
            'concept_scores': {'general': 0.5},
            'primary_concept': 'general',
            'semantic_complexity': medical_count / len(words) if words else 0.0,
            'embedding_dimension': 5  # Fallback feature dimension
        }
    
    def smart_context_selection_with_medembed(self, query: str, context_docs: List[Dict]) -> List[Dict]:
        """Use MedEmbed to select most relevant context documents"""
        if not context_docs or not self._embedder_ready:
            return context_docs[:self.max_context_docs]
        
        print(f"🔍 Using MedEmbed for intelligent context selection from {len(context_docs)} documents...")
        
        # Generate embeddings
        query_embedding = self._embed_text([query])[0]
        context_texts = [doc.get('text', '')[:500] for doc in context_docs]
        context_embeddings = self._embed_text(context_texts)
        
        # Calculate relevance scores
        scored_docs = []
        for i, doc in enumerate(context_docs):
            if i < len(context_embeddings):
                similarity = self._cosine_similarity(query_embedding, context_embeddings[i])
                scored_docs.append((similarity, doc))
        
        # Sort by relevance and return top documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        selected_docs = [doc for score, doc in scored_docs[:self.max_context_docs]]
        
        scores = [score for score, doc in scored_docs[:self.max_context_docs]]
        print(f"✅ Selected {len(selected_docs)} most relevant documents (scores: {[f'{s:.3f}' for s in scores]})")
        
        return selected_docs
    
    def detect_medical_specialty_with_medembed(self, text: str, context_docs: List[Dict] = None) -> MedicalSpecialty:
        """Enhanced specialty detection using MedEmbed embeddings"""
        if not self._embedder_ready:
            return self.detect_medical_specialty(text)  # Fallback to keyword method
        
        # Combine text with context for better specialty detection
        combined_text = text
        if context_docs:
            context_text = " ".join([doc.get('text', '')[:200] for doc in context_docs[:2]])
            combined_text += " " + context_text
        
        # Generate embedding for combined text
        text_embedding = self._embed_text([combined_text])[0]
        
        # Specialty prototype embeddings
        specialty_descriptions = {
            MedicalSpecialty.CARDIOLOGY: "cardiovascular heart cardiac chest pain arrhythmia blood pressure coronary",
            MedicalSpecialty.ENDOCRINOLOGY: "diabetes thyroid hormone insulin glucose endocrine metabolic",
            MedicalSpecialty.PULMONOLOGY: "lung respiratory breathing pneumonia asthma cough dyspnea",
            MedicalSpecialty.NEUROLOGY: "brain neurological seizure stroke headache weakness numbness",
            MedicalSpecialty.GASTROENTEROLOGY: "stomach digestive abdominal liver gastric intestinal nausea"
        }
        
        specialty_embeddings = self._embed_text(list(specialty_descriptions.values()))
        
        # Calculate similarities
        best_specialty = MedicalSpecialty.GENERAL_MEDICINE
        best_score = 0.0
        
        for i, (specialty, description) in enumerate(specialty_descriptions.items()):
            similarity = self._cosine_similarity(text_embedding, specialty_embeddings[i])
            if similarity > best_score:
                best_score = similarity
                best_specialty = specialty
        
        print(f"🎯 MedEmbed specialty detection: {best_specialty.value} (confidence: {best_score:.3f})")
        
        return best_specialty if best_score > 0.4 else MedicalSpecialty.GENERAL_MEDICINE
    
    # ===============================================================================
    # ✅ ENERGY CALCULATION METHODS (UPDATED FOR KWH & ROUNDED VALUES)
    # ===============================================================================
    
    def calculate_enhanced_tokens(self, text: str) -> int:
        """Enhanced token estimation with medical complexity factors"""
        if not text:
            return 0
        
        # Method 1: Word count with coefficient
        word_count = len(text.split())
        word_estimate = int(word_count * 1.2)  # Conservative estimate
        
        # Method 2: Character-based estimation
        char_estimate = int(len(text) / 4.5)
        
        # Method 3: Medical complexity adjustment (minimal)
        medical_indicators = ['patient', 'medical', 'clinical', 'diagnosis']
        medical_count = sum(1 for indicator in medical_indicators if indicator in text.lower())
        complexity_multiplier = 1.0 + (medical_count * 0.02)  # Small adjustment
        
        # Weighted ensemble
        base_estimate = (word_estimate * 0.7 + char_estimate * 0.3)
        final_estimate = int(base_estimate * complexity_multiplier)
        
        return max(1, final_estimate)
    
    def calculate_energy_metrics(self, original_text: str, optimized_text: str, 
                               processing_time_ms: float = 0) -> EnergyMetrics:
        """✅ UPDATED: Calculate energy metrics with kWh and rounded values"""
        original_tokens = self.calculate_enhanced_tokens(original_text)
        optimized_tokens = self.calculate_enhanced_tokens(optimized_text)
        tokens_reduced = max(0, original_tokens - optimized_tokens)
        
        # Energy calculations in Wh first
        base_energy_saved_wh = tokens_reduced * self.ENERGY_PER_TOKEN
        processing_energy_wh = (processing_time_ms / 1000) * self.PROCESSING_OVERHEAD
        net_energy_saved_wh = max(0, base_energy_saved_wh - processing_energy_wh)
        
        # ✅ CONVERT TO KWH
        net_energy_saved_kwh = net_energy_saved_wh / 1000  # Convert Wh to kWh
        
        # Environmental impact (using kWh)
        co2_saved = net_energy_saved_kwh * self.CO2_PER_KWH
        cost_saved = net_energy_saved_kwh * self.COST_PER_KWH
        
        # ✅ ROUNDED: Efficiency metrics with proper rounding
        efficiency_improvement = round((tokens_reduced / original_tokens * 100), 1) if original_tokens > 0 else 0.0
        compression_ratio = round((optimized_tokens / original_tokens), 3) if original_tokens > 0 else 1.0
        
        return EnergyMetrics(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            tokens_reduced=tokens_reduced,
            energy_saved_kwh=round(net_energy_saved_kwh, 9),    # ✅ kWh with 9 decimal places
            cost_saved_usd=round(cost_saved, 8),                # ✅ Rounded to 8 decimal places
            co2_reduced_kg=round(co2_saved, 8),                 # ✅ Rounded to 8 decimal places
            efficiency_improvement_percent=efficiency_improvement,  # ✅ Already rounded to 1 decimal
            processing_time_ms=round(processing_time_ms, 1),
            compression_ratio=compression_ratio                 # ✅ Already rounded to 3 decimal places
        )
    
    # ===============================================================================
    # ✅ COMPRESSION METHODS
    # ===============================================================================
    
    def aggressive_compress_medical_text(self, text: str) -> Tuple[str, List[str]]:
        """Aggressively compress medical text while preserving meaning"""
        compressed = text.strip()
        techniques_used = []
        
        # Stage 1: Remove conversational redundancy
        redundant_patterns = [
            r'\bplease\s+', r'\bcan\s+you\s+', r'\bcould\s+you\s+please\s*',
            r'\bi\s+would\s+like\s+to\s+know\s+', r'\bi\s+need\s+help\s+with\s+',
            r'\bhelp\s+me\s+understand\s+', r'\btell\s+me\s+about\s+',
            r'\bexplain\s+to\s+me\s+', r'\blet\s+me\s+know\s+about\s+',
            r'\bprovide\s+information\s+(about|on)\s+', r'\bkindly\s+'
        ]
        
        for pattern in redundant_patterns:
            if re.search(pattern, compressed, re.IGNORECASE):
                compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)
                techniques_used.append("redundancy_removal")
        
        # Stage 2: Apply medical abbreviations
        for pattern, replacement in self.medical_compressions.items():
            if re.search(pattern, compressed, re.IGNORECASE):
                compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
                techniques_used.append("medical_abbreviations")
        
        # Stage 3: Sentence structure compression
        structure_compressions = {
            r'\s+with\s+a\s+': ' w/ ',
            r'\s+and\s+also\s+': ' & ',
            r'\bpresenting\s+with\s+': 'w/ ',
            r'\bcomplaining\s+of\s+': 'c/o ',
            r'\bhas\s+a\s+history\s+of\s+': 'h/o ',
            r'\bmedical\s+history\s+includes\s+': 'PMH: '
        }
        
        for pattern, replacement in structure_compressions.items():
            if re.search(pattern, compressed, re.IGNORECASE):
                compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
                techniques_used.append("structure_compression")
        
        # Clean up whitespace
        compressed = re.sub(r'\s+', ' ', compressed).strip()
        
        return compressed, list(set(techniques_used))
    
    def smart_medical_truncation(self, text: str, max_length: int) -> str:
        """Smart truncation that preserves important medical information"""
        if len(text) <= max_length:
            return text
        
        # Preserve critical medical sections
        critical_sections = [
            r'##?\s*(ASSESSMENT|DIAGNOSIS|PLAN|FINDINGS|VITALS?|MEDICATIONS?)',
            r'(EKG|ECG|CT|MRI|X-ray|Blood pressure|Heart rate|Temperature)',
            r'(\d+/\d+\s*mmHg|\d+\s*bpm|\d+\s*°F|\d+\s*°C)',  # Vital signs
            r'(mg|mcg|units|tablets?)\b',  # Medication dosages
        ]
        
        # Find critical medical content
        critical_content = []
        for pattern in critical_sections:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                critical_content.append(text[start:end])
        
        if critical_content:
            # Preserve critical content and truncate the rest
            preserved = " | ".join(critical_content)
            if len(preserved) <= max_length - 50:
                remaining_space = max_length - len(preserved) - 50
                beginning = text[:remaining_space]
                return f"{beginning}...\n\nCRITICAL: {preserved}"
        
        # Fallback: Truncate at sentence boundary
        truncation_point = max_length - 20
        sentences = text[:truncation_point].split('.')
        if len(sentences) > 1:
            truncated = '.'.join(sentences[:-1]) + '.'
            return truncated + "..."
        
        return text[:truncation_point] + "..."
    
    def extract_critical_context(self, context_docs: List[Dict]) -> str:
        """Extract critical information from context documents"""
        critical_info = []
        for doc in context_docs[:2]:
            text = doc.get('text', '')
            # Look for vital signs, lab values, etc.
            vitals = re.findall(r'\d+/\d+\s*mmHg|\d+\s*bpm|\d+\s*°[FC]', text)
            labs = re.findall(r'\w+:\s*\d+\.?\d*', text)
            if vitals:
                critical_info.extend(vitals[:2])
            if labs:
                critical_info.extend(labs[:2])
        
        return ", ".join(critical_info)
    
    # ===============================================================================
    # ✅ CORE OPTIMIZATION METHODS
    # ===============================================================================
    
    def extract_patient_info(self, text: str) -> PatientInfo:
        """Enhanced patient information extraction"""
        patient_info = PatientInfo()
        
        # Enhanced name patterns
        name_patterns = [
            r'Patient Name:\s*([^,\n]+)',
            r'Patient:\s*([^,\n]+)', 
            r'Name is\s*([A-Z][^\n,]+)',
            r'case.*?([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+),\s*(\d+)\s*(?:years?\s*old|yo)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                name = re.sub(r'^(Patient|pt|Mr|Mrs|Ms)\.?\s*', '', name, flags=re.IGNORECASE)
                if len(name) > 2 and not name.lower() in ['male', 'female', 'patient']:
                    patient_info.name = name
                    break
        
        # Enhanced age patterns
        age_patterns = [
            r'(\d+)\s*(?:year[s]?\s*old|yo|y/o)',
            r'Age:\s*(\d+)',
            r'age\s*(\d+)',
            r'(\d+)-year-old'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                patient_info.age = match.group(1)
                break
        
        # Enhanced gender patterns
        gender_patterns = [
            r'(\d+)\s*(?:year[s]?\s*old|yo|y/o)\s*(male|female)',
            r'(male|female)\s*patient',
            r'Gender:\s*(male|female|M|F)\b'
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender = match.group(-1).lower()
                patient_info.gender = "M" if gender in ['male', 'm'] else "F"
                break
        
        # Extract chief complaint
        cc_patterns = [
            r'presenting with\s*([^.]+)',
            r'complaining of\s*([^.]+)',
            r'Chief Complaint:\s*([^\n]+)'
        ]
        
        for pattern in cc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                patient_info.chief_complaint = match.group(1).strip()
                break
        
        return patient_info
    
    def classify_query_type(self, query: str) -> QueryType:
        """Classify query type using keywords"""
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in ['report', 'comprehensive', 'medical record']):
            return QueryType.PATIENT_REPORT
        elif any(indicator in query_lower for indicator in ['symptoms', 'presenting', 'complaint']):
            return QueryType.SYMPTOM_ANALYSIS
        elif any(indicator in query_lower for indicator in ['treatment', 'therapy', 'management']):
            return QueryType.TREATMENT_PLAN
        elif any(indicator in query_lower for indicator in ['diagnosis', 'workup']):
            return QueryType.DIAGNOSTIC_WORKUP
        else:
            return QueryType.GENERAL_MEDICAL
    
    def detect_medical_specialty(self, text: str) -> MedicalSpecialty:
        """Fallback specialty detection using keywords"""
        text_lower = text.lower()
        specialty_scores = {}
        
        for specialty, keywords in self.specialty_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                specialty_scores[specialty] = score
        
        return max(specialty_scores, key=specialty_scores.get) if specialty_scores else MedicalSpecialty.GENERAL_MEDICINE
    
    def summarize_context(self, context_docs: List[Dict], max_chars: int = 1200) -> str:
        """Create context summary with more space for medical content"""
        if not context_docs:
            return "No context."
        
        summaries = []
        char_count = 0
        
        for doc in context_docs[:self.max_context_docs]:
            text = doc.get('text', '')[:400]
            summary = text.replace('\n', ' ')
            
            if char_count + len(summary) > max_chars:
                remaining_space = max_chars - char_count
                if remaining_space > 200:
                    summary = summary[:remaining_space - 10] + "..."
                    summaries.append(summary)
                break
            
            summaries.append(summary)
            char_count += len(summary)
        
        return " | ".join(summaries)
    
    def medembed_guided_optimization(self, query: str, context_docs: List[Dict], 
                                    semantic_analysis: Dict, patient_info) -> str:
        """MedEmbed-guided intelligent optimization"""
        
        primary_concept = semantic_analysis.get('primary_concept', 'general')
        semantic_complexity = semantic_analysis.get('semantic_complexity', 0.5)
        
        print(f"🎯 Applying MedEmbed-guided optimization for {primary_concept} concept")
        
        # Start with aggressive compression
        optimized, techniques = self.aggressive_compress_medical_text(query)
        
        # Concept-specific optimization strategies
        if primary_concept == 'emergency':
            # For emergency cases, prioritize speed and critical info
            optimized = f"URGENT: {optimized}"
            if context_docs:
                critical_context = self.extract_critical_context(context_docs)
                if critical_context:
                    optimized += f" | Critical: {critical_context[:100]}"
        
        elif primary_concept == 'diagnostic':
            # For diagnostic queries, emphasize systematic approach
            if patient_info.name:
                optimized = f"{patient_info.name}: {optimized} - systematic diagnostic approach needed"
            else:
                optimized = f"Diagnostic evaluation: {optimized}"
        
        elif primary_concept == 'therapeutic':
            # For treatment queries, focus on actionable plans
            optimized = f"Treatment plan: {optimized}"
        
        elif semantic_complexity > 0.7:
            # High complexity queries - preserve more context
            if context_docs:
                relevant_context = self.summarize_context(context_docs, max_chars=200)
                optimized += f" | Context: {relevant_context}"
        
        return optimized
    
    # ===============================================================================
    # ✅ MAIN OPTIMIZATION METHOD
    # ===============================================================================
    
    def optimize_prompt(self, query: str, context_docs: List[Dict] = None) -> OptimizationResult:
        """Enhanced optimization using MedEmbed semantic understanding"""
        start_time = datetime.now()
        
        try:
            if context_docs is None:
                context_docs = []
            
            # Clear torch cache if using GPU
            if TORCH_AVAILABLE and hasattr(self, '_device') and self._device.type != 'cpu':
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # ✅ USE MEDEMBED: Analyze query semantically
            semantic_analysis = self.analyze_query_with_medembed(query, context_docs)
            primary_concept = semantic_analysis.get('primary_concept', 'general')
            semantic_complexity = semantic_analysis.get('semantic_complexity', 0.5)
            
            print(f"🧠 MedEmbed analysis: Primary concept = {primary_concept}, Complexity = {semantic_complexity:.3f}")
            
            # ✅ USE MEDEMBED: Smart context selection
            if context_docs:
                context_docs = self.smart_context_selection_with_medembed(query, context_docs)
            
            # Extract basic information
            patient_info = self.extract_patient_info(query)
            query_type = self.classify_query_type(query)
            specialty = self.detect_medical_specialty_with_medembed(query, context_docs)
            
            # ✅ ADAPTIVE OPTIMIZATION: Based on MedEmbed analysis
            if self.energy_mode:
                # Use semantic understanding to guide optimization
                optimized_prompt = self.medembed_guided_optimization(
                    query, 
                    context_docs, 
                    semantic_analysis,
                    patient_info
                )
                techniques_used = ["medembed_semantic_optimization"]
            else:
                # Standard template-based approach
                template = self.templates.get(query_type, self.templates[QueryType.GENERAL_MEDICAL])
                context_summary = self.summarize_context(context_docs, max_chars=800)
                
                template_vars = {
                    'patient_name': patient_info.name or 'Patient',
                    'patient_age': patient_info.age or '?',
                    'patient_gender': patient_info.gender or '?',
                    'mrn': patient_info.mrn or '',
                    'chief_complaint': patient_info.chief_complaint or query[:100],
                    'context_summary': context_summary or 'None',
                    'symptom_description': query if query_type == QueryType.SYMPTOM_ANALYSIS else '',
                    'condition_description': query if query_type == QueryType.TREATMENT_PLAN else '',
                    'query_text': query
                }
                
                optimized_prompt = template.format(**template_vars)
                techniques_used = ["template_with_medembed_context"]
            
            # Smart truncation if needed
            if len(optimized_prompt) > self.max_prompt_length:
                optimized_prompt = self.smart_medical_truncation(optimized_prompt, self.max_prompt_length)
                techniques_used.append("smart_truncation")
            
            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            energy_metrics = self.calculate_energy_metrics(query, optimized_prompt, processing_time)
            
            metrics = PromptMetrics(
                length=len(optimized_prompt),
                token_estimate=energy_metrics.optimized_tokens,
                context_utilization=min(len(context_docs) / max(self.max_context_docs, 1), 1.0),
                patient_specificity=0.8 if patient_info.name else 0.3,
                medical_terminology_density=semantic_complexity
            )
            
            return OptimizationResult(
                optimized_prompt=optimized_prompt,
                original_prompt=query,
                energy_metrics=energy_metrics,
                query_type=query_type.value,
                medical_specialty=specialty.value,
                patient_info=patient_info,
                metrics=metrics,
                language='English',
                timestamp=datetime.now().isoformat(),
                compression_techniques_used=techniques_used
            )
            
        except Exception as e:
            print(f"❌ MedEmbed optimization error: {e}")
            return self._fallback_optimization(query, context_docs)
    
    def _fallback_optimization(self, query: str, context_docs: List[Dict]) -> OptimizationResult:
        """Fallback optimization when MedEmbed fails"""
        compressed_query, techniques = self.aggressive_compress_medical_text(query)
        energy_metrics = self.calculate_energy_metrics(query, compressed_query)
        
        return OptimizationResult(
            optimized_prompt=compressed_query,
            original_prompt=query,
            energy_metrics=energy_metrics,
            query_type='general_medical',
            medical_specialty='general_medicine',
            patient_info=PatientInfo(),
            metrics=PromptMetrics(len(compressed_query), energy_metrics.optimized_tokens, 0.0, 0.0, 0.5),
            language='English',
            timestamp=datetime.now().isoformat(),
            compression_techniques_used=techniques
        )

# ===============================================================================
# ✅ FLASK INTEGRATION WRAPPER (ENHANCED)
# ===============================================================================

class FlaskIntegrationWrapper:
    """Enhanced wrapper with MedEmbed integration for Flask application"""
    
    def __init__(self):
        self.optimizer = EnergyEfficientMedicalOptimizer(energy_mode=True)
        print("✅ FlaskIntegrationWrapper initialized with MedEmbed-enhanced EnergyEfficientMedicalOptimizer")
    
    def optimize_for_flask(self, query: str, context_docs: List[Dict] = None) -> Dict[str, Any]:
        """Optimize and return Flask-compatible response with MedEmbed insights"""
        result = self.optimizer.optimize_prompt(query, context_docs or [])
        
        return {
            'success': True,
            'optimized_prompt': result.optimized_prompt,
            'original_query': result.original_prompt,
            'energy_metrics': {
                'tokens_reduced': result.energy_metrics.tokens_reduced,
                'energy_saved_kwh': result.energy_metrics.energy_saved_kwh,    # ✅ Now in kWh
                'efficiency_improvement_percent': result.energy_metrics.efficiency_improvement_percent,
                'cost_saved_usd': result.energy_metrics.cost_saved_usd,
                'original_tokens': result.energy_metrics.original_tokens,
                'optimized_tokens': result.energy_metrics.optimized_tokens,
                'compression_ratio': result.energy_metrics.compression_ratio
            },
            'original_metrics': {
                'length': len(result.original_prompt),
                'token_estimate': result.energy_metrics.original_tokens,
                'medical_terminology_density': 0.5
            },
            'metrics': {
                'length': result.metrics.length,
                'token_estimate': result.metrics.token_estimate,
                'medical_terminology_density': result.metrics.medical_terminology_density
            },
            'enhanced_analysis': {
                'query_type': result.query_type,
                'medical_specialty': result.medical_specialty,
                'compression_techniques_used': result.compression_techniques_used,
                'patient_info_extracted': bool(result.patient_info.name),
                'medembed_powered': self.optimizer._embedder_ready,
                'semantic_optimization': True,
                'energy_optimization_version': '3.0_medembed_kwh_rounded'
            }
        }
    
    def optimize_prompt(self, query: str, context_docs: List[Dict] = None) -> Dict[str, Any]:
        """Enhanced RAG-compatible optimization using MedEmbed"""
        print(f"🧠 MedEmbed-powered RAG optimization for: {query[:50]}...")
        
        flask_result = self.optimize_for_flask(query, context_docs)
        
        # Convert to RAG-compatible format with MedEmbed insights
        return {
            'optimized_prompt': flask_result['optimized_prompt'],
            'query_type': flask_result['enhanced_analysis']['query_type'],
            'medical_specialty': flask_result['enhanced_analysis']['medical_specialty'],
            'patient_info': {
                'name': None,
                'age': None,
                'gender': None,
                'chief_complaint': query[:100] if query else None
            },
            'metrics': flask_result['metrics'],
            'energy_analysis': flask_result['energy_metrics'],
            'medembed_analysis': {
                'semantic_optimization': True,
                'embedding_powered': flask_result['enhanced_analysis']['medembed_powered'],
                'techniques_used': flask_result['enhanced_analysis']['compression_techniques_used']
            },
            'language': 'English',
            'timestamp': datetime.now().isoformat(),
            'success': True
        }

# Backward compatibility aliases
EnglishMedicalPromptOptimizer = FlaskIntegrationWrapper

# Test function
def test_medembed_integration():
    """Test the MedEmbed-enhanced optimizer"""
    print("🧪 Testing MedEmbed-Enhanced Medical Optimizer...")
    
    optimizer = EnergyEfficientMedicalOptimizer(energy_mode=True)
    
    test_query = "Patient Maria Rodriguez, 67 years old Hispanic female presenting to ED with c/o SOB and chest discomfort for the past 2 hours. Medical history includes hypertension and diabetes."
    
    result = optimizer.optimize_prompt(test_query)
    
    print(f"\n✅ MedEmbed Test Results:")
    print(f"   Original: {len(test_query)} chars, {result.energy_metrics.original_tokens} tokens")
    print(f"   Optimized: {len(result.optimized_prompt)} chars, {result.energy_metrics.optimized_tokens} tokens")
    print(f"   Tokens Saved: {result.energy_metrics.tokens_reduced}")
    print(f"   Energy Saved: {result.energy_metrics.energy_saved_kwh:.9f} kWh")  # ✅ Now shows kWh
    print(f"   Efficiency: {result.energy_metrics.efficiency_improvement_percent}%")  # ✅ Now rounded
    print(f"   Specialty: {result.medical_specialty}")
    print(f"   Techniques: {result.compression_techniques_used}")
    print(f"   MedEmbed Active: {optimizer._embedder_ready}")
    
    return True

if __name__ == "__main__":
    test_medembed_integration()
