"""
English-Focused Medical Prompt Optimizer - Milestone 5 Simplified
Streamlined prompt optimization for English medical content
"""

import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    """Medical query types for English content"""
    PATIENT_REPORT = "patient_report"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    TREATMENT_PLAN = "treatment_plan"
    DIAGNOSTIC_WORKUP = "diagnostic_workup"
    MEDICATION_REVIEW = "medication_review"
    FOLLOW_UP = "follow_up"
    GENERAL_MEDICAL = "general_medical"

class MedicalSpecialty(Enum):
    """Medical specialties for English content"""
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
    """Patient information extraction"""
    name: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None
    mrn: Optional[str] = None
    chief_complaint: Optional[str] = None

@dataclass
class PromptMetrics:
    """Prompt quality metrics"""
    length: int
    token_estimate: int
    context_utilization: float
    patient_specificity: float
    medical_terminology_density: float

class EnglishMedicalPromptOptimizer:
    """
    Streamlined English-focused medical prompt optimizer
    """
    
    def __init__(self, max_prompt_length: int = 2000, max_context_docs: int = 3):
        self.max_prompt_length = max_prompt_length
        self.max_context_docs = max_context_docs
        self.prompt_cache = {}
        
        # ✅ ENHANCED: Comprehensive English medical terminology
        # Based on common medical usage patterns and terminology
        self.specialty_keywords = {
            MedicalSpecialty.CARDIOLOGY: [
                # Core cardiac terms
                'heart', 'cardiac', 'cardiovascular', 'myocardial', 'coronary',
                # Diagnostic terms
                'ecg', 'ekg', 'echo', 'echocardiogram', 'angiogram', 'catheterization',
                # Conditions
                'arrhythmia', 'tachycardia', 'bradycardia', 'hypertension', 'hypotension',
                # Symptoms
                'chest pain', 'palpitations', 'shortness of breath', 'dyspnea',
                # Treatments
                'angioplasty', 'stent', 'bypass', 'pacemaker', 'defibrillator'
            ],
            
            MedicalSpecialty.ENDOCRINOLOGY: [
                # Core endocrine terms
                'diabetes', 'thyroid', 'hormone', 'endocrine', 'metabolic',
                # Diagnostic terms
                'glucose', 'insulin', 'hba1c', 'tsh', 'cortisol', 'hormone levels',
                # Conditions
                'hyperthyroid', 'hypothyroid', 'adrenal', 'pituitary',
                # Medications
                'metformin', 'insulin', 'glipizide', 'levothyroxine'
            ],
            
            MedicalSpecialty.PULMONOLOGY: [
                # Core respiratory terms
                'lung', 'respiratory', 'pulmonary', 'bronchial', 'alveolar',
                # Conditions
                'asthma', 'copd', 'pneumonia', 'bronchitis', 'emphysema',
                # Symptoms
                'cough', 'sputum', 'wheezing', 'dyspnea',
                # Diagnostics
                'chest x-ray', 'ct chest', 'pulmonary function', 'spirometry'
            ],
            
            MedicalSpecialty.NEUROLOGY: [
                # Core neurological terms
                'brain', 'neurological', 'neural', 'cerebral', 'spinal',
                # Conditions
                'seizure', 'epilepsy', 'stroke', 'migraine', 'dementia',
                # Symptoms
                'headache', 'dizziness', 'numbness', 'weakness', 'tremor',
                # Diagnostics
                'mri brain', 'ct head', 'eeg', 'lumbar puncture'
            ],
            
            MedicalSpecialty.GASTROENTEROLOGY: [
                # Core GI terms
                'stomach', 'intestinal', 'digestive', 'gastrointestinal', 'abdominal',
                # Organs
                'liver', 'gallbladder', 'pancreas', 'colon', 'esophagus',
                # Conditions
                'ulcer', 'reflux', 'hepatitis', 'cirrhosis', 'pancreatitis',
                # Symptoms
                'nausea', 'vomiting', 'diarrhea', 'constipation', 'abdominal pain'
            ],
            
            MedicalSpecialty.ORTHOPEDICS: [
                # Core orthopedic terms
                'bone', 'joint', 'orthopedic', 'musculoskeletal', 'skeletal',
                # Conditions
                'fracture', 'arthritis', 'osteoporosis', 'tendonitis',
                # Body parts
                'spine', 'knee', 'shoulder', 'hip', 'ankle', 'wrist',
                # Treatments
                'surgery', 'physical therapy', 'cast', 'splint'
            ],
            
            MedicalSpecialty.ONCOLOGY: [
                # Core cancer terms
                'cancer', 'tumor', 'oncology', 'malignant', 'benign', 'metastasis',
                # Treatments
                'chemotherapy', 'radiation', 'immunotherapy', 'surgery',
                # Diagnostics
                'biopsy', 'ct scan', 'pet scan', 'staging'
            ],
            
            MedicalSpecialty.PSYCHIATRY: [
                # Core psychiatric terms
                'mental health', 'psychiatric', 'psychological', 'behavioral',
                # Conditions
                'depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd',
                # Treatments
                'antidepressant', 'therapy', 'counseling', 'psychotherapy'
            ],
            
            MedicalSpecialty.DERMATOLOGY: [
                # Core dermatologic terms
                'skin', 'dermatology', 'rash', 'lesion', 'dermatitis',
                # Conditions
                'eczema', 'psoriasis', 'acne', 'melanoma', 'basal cell'
            ]
        }
        
        # Medical terminology patterns for English
        self.medical_patterns = {
            'vital_signs': r'(?:BP|blood pressure|heart rate|HR|temperature|temp|respiratory rate|RR|oxygen saturation|O2 sat)\s*:?\s*[\d\/\-]+',
            'medications': r'(?:mg|mcg|units|tablets?|capsules?|ml|cc|bid|tid|qid|prn)\b',
            'lab_values': r'(?:WBC|RBC|Hgb|Hct|PLT|glucose|BUN|creatinine|sodium|potassium|chloride)\s*:?\s*[\d\.]+',
            'medical_procedures': r'(?:CT|MRI|X-ray|ultrasound|EKG|ECG|biopsy|surgery)\b'
        }
        
        # Initialize English prompt templates
        self._init_prompt_templates()
        
        print("✅ English Medical Prompt Optimizer initialized")
    
    def _init_prompt_templates(self):
        """Initialize medical prompt templates for English"""
        self.templates = {
            QueryType.PATIENT_REPORT: """You are a medical AI assistant. Generate a comprehensive medical report based on the information provided.

PATIENT INFORMATION:
{patient_info}

RELEVANT MEDICAL RECORDS:
{context_summary}

Generate a detailed medical report with the following sections:

## COMPREHENSIVE MEDICAL REPORT: {patient_name}

### PATIENT IDENTIFICATION
- Patient Name: {patient_name}
- Date of Report: {current_date}
- Medical Record Number: {mrn}

### CHIEF COMPLAINT & HISTORY
Detail the patient's primary concerns and relevant medical history.

### CURRENT MEDICATIONS & ALLERGIES
List medications, dosages, and known allergies.

### CLINICAL FINDINGS
Include vital signs, examination findings, and test results.

### ASSESSMENT & DIAGNOSIS
Primary and secondary diagnoses with clinical reasoning.

### TREATMENT PLAN
Detailed treatment recommendations and follow-up care.

Generate the medical report using the provided information:""",

            QueryType.SYMPTOM_ANALYSIS: """You are a medical AI assistant specializing in symptom analysis.

PATIENT SYMPTOMS:
{symptom_description}

CLINICAL CONTEXT:
{context_summary}

Provide a comprehensive symptom analysis including:

## SYMPTOM ANALYSIS

### SYMPTOM CHARACTERIZATION
Detail the symptoms with onset, duration, and characteristics.

### DIFFERENTIAL DIAGNOSIS
List potential diagnoses ranked by likelihood with supporting evidence.

### RECOMMENDED DIAGNOSTIC WORKUP
Suggest appropriate tests, imaging, and consultations.

### MANAGEMENT RECOMMENDATIONS
Provide treatment suggestions and monitoring plans.

Generate the analysis:""",

            QueryType.TREATMENT_PLAN: """You are a medical AI assistant specializing in treatment planning.

MEDICAL CONDITION:
{condition_description}

CLINICAL INFORMATION:
{context_summary}

Develop a comprehensive treatment plan:

## TREATMENT PLAN

### THERAPEUTIC GOALS
Primary and secondary treatment objectives.

### PHARMACOLOGICAL MANAGEMENT
Medications with dosing and monitoring requirements.

### NON-PHARMACOLOGICAL INTERVENTIONS
Lifestyle modifications and supportive care.

### FOLLOW-UP & MONITORING
Surveillance plan and outcome measures.

Generate the treatment plan:""",

            QueryType.GENERAL_MEDICAL: """You are a medical AI assistant. Please provide a comprehensive medical response.

MEDICAL QUERY:
{symptom_description}{condition_description}

CLINICAL CONTEXT:
{context_summary}

Provide a detailed medical response addressing the query:"""
        }
    
    def extract_patient_info(self, text: str) -> PatientInfo:
        """Extract patient information from English text"""
        patient_info = PatientInfo()
        
        # Patient name patterns
        name_patterns = [
            r'Patient Name:\s*([^,\n]+)',
            r'Patient:\s*([^,\n]+)',
            r'Name:\s*([A-Z][a-zA-Z\'\-\s]+)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                patient_info.name = match.group(1).strip()
                break
        
        # Age extraction
        age_patterns = [
            r'Age:\s*(\d+)',
            r'(\d+)\s*(?:year[s]?\s*old|yo|y/o)',
            r'age\s*(\d+)'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                patient_info.age = match.group(1)
                break
        
        # Gender extraction
        gender_patterns = [
            r'Gender:\s*(male|female|M|F)\b',
            r'\b(male|female)\s*patient'
        ]
        
        for pattern in gender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                gender = match.group(1).lower()
                patient_info.gender = "Male" if gender in ['male', 'm'] else "Female"
                break
        
        # MRN extraction
        mrn_patterns = [
            r'MRN:\s*([A-Z0-9\-]+)',
            r'Medical Record Number:\s*([A-Z0-9\-]+)'
        ]
        
        for pattern in mrn_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                patient_info.mrn = match.group(1)
                break
        
        # Chief complaint
        cc_patterns = [
            r'Chief Complaint:\s*([^\n]+)',
            r'CC:\s*([^\n]+)',
            r'presents with\s*([^\n]+)'
        ]
        
        for pattern in cc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                patient_info.chief_complaint = match.group(1).strip()
                break
        
        return patient_info
    
    def classify_query_type(self, query: str) -> QueryType:
        """Classify the type of medical query"""
        query_lower = query.lower()
        
        if any(indicator in query_lower for indicator in ['patient name:', 'medical record', 'comprehensive report']):
            return QueryType.PATIENT_REPORT
        elif any(indicator in query_lower for indicator in ['symptoms', 'presenting with', 'chief complaint']):
            return QueryType.SYMPTOM_ANALYSIS
        elif any(indicator in query_lower for indicator in ['treatment', 'therapy', 'management']):
            return QueryType.TREATMENT_PLAN
        elif any(indicator in query_lower for indicator in ['diagnosis', 'diagnostic', 'workup']):
            return QueryType.DIAGNOSTIC_WORKUP
        else:
            return QueryType.GENERAL_MEDICAL
    
    def detect_medical_specialty(self, text: str) -> MedicalSpecialty:
        """Detect medical specialty from English text"""
        text_lower = text.lower()
        specialty_scores = {}
        
        for specialty, keywords in self.specialty_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                specialty_scores[specialty] = score
        
        if specialty_scores:
            return max(specialty_scores, key=specialty_scores.get)
        else:
            return MedicalSpecialty.GENERAL_MEDICINE
    
    def summarize_context(self, context_docs: List[Dict], max_chars: int = 1000) -> str:
        """Summarize context documents for English content"""
        if not context_docs:
            return "No relevant medical documents found."
        
        # Filter by relevance
        relevant_docs = [doc for doc in context_docs if doc.get('similarity', 0) > 0.3]
        if not relevant_docs:
            relevant_docs = context_docs[:self.max_context_docs]
        
        summaries = []
        total_chars = 0
        
        for i, doc in enumerate(relevant_docs[:self.max_context_docs]):
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # Extract meaningful content
            summary = text[:400].replace('\n', ' ')
            
            if total_chars + len(summary) > max_chars:
                remaining_chars = max_chars - total_chars
                if remaining_chars > 100:
                    summary = summary[:remaining_chars] + "..."
                else:
                    break
            
            file_name = metadata.get('file_name', f'Document {i+1}')
            formatted_summary = f"=== {file_name} ===\n{summary}\n"
            summaries.append(formatted_summary)
            total_chars += len(formatted_summary)
        
        return "\n".join(summaries)
    
    def optimize_prompt(self, query: str, context_docs: List[Dict]) -> Dict[str, Any]:
        """
        Main prompt optimization for English medical content
        """
        try:
            print(f"🔄 Optimizing English medical prompt: {query[:50]}...")
            
            # Extract patient information
            patient_info = self.extract_patient_info(query)
            
            # Classify query and specialty
            query_type = self.classify_query_type(query)
            specialty = self.detect_medical_specialty(query + ' '.join([doc.get('text', '') for doc in context_docs[:2]]))
            
            print(f"👤 Patient: {patient_info.name or 'Not specified'}")
            print(f"🔍 Query type: {query_type.value}")  # ✅ FIX: Use .value
            print(f"🏥 Specialty: {specialty.value}")     # ✅ FIX: Use .value
            
            # Get template - ✅ FIX: Handle template selection safely
            if query_type in self.templates:
                template = self.templates[query_type]
            else:
                template = self.templates[QueryType.GENERAL_MEDICAL]
            
            # Summarize context
            context_summary = self.summarize_context(context_docs)
            
            # Format template
            template_vars = {
                'patient_name': patient_info.name or 'Patient',
                'patient_info': self._format_patient_info(patient_info),
                'context_summary': context_summary,
                'current_date': datetime.now().strftime('%Y-%m-%d'),
                'mrn': patient_info.mrn or 'Not documented',
                'symptom_description': query if query_type == QueryType.SYMPTOM_ANALYSIS else '',
                'condition_description': query if query_type == QueryType.TREATMENT_PLAN else ''
            }
            
            # ✅ FIX: Safe template formatting
            try:
                optimized_prompt = template.format(**template_vars)
            except KeyError as template_error:
                print(f"⚠️ Template formatting error: {template_error}")
                # Simple fallback prompt
                optimized_prompt = f"""You are a medical AI assistant.

Patient Query: {query}

Relevant Medical Information:
{context_summary}

Please provide a comprehensive medical response:"""
            
            # Ensure length limits
            if len(optimized_prompt) > self.max_prompt_length:
                optimized_prompt = optimized_prompt[:self.max_prompt_length - 50] + "\n[Truncated]"
            
            # Calculate metrics
            metrics = self._calculate_metrics(optimized_prompt, context_docs, patient_info)
            
            # ✅ FIX: Ensure all enum values are converted to strings
            result = {
                'optimized_prompt': optimized_prompt,
                'query_type': query_type.value,        # ✅ Convert enum to string
                'medical_specialty': specialty.value,  # ✅ Convert enum to string
                'patient_info': patient_info,
                'metrics': metrics,
                'language': 'English',
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"✅ Optimization complete: {len(optimized_prompt)} chars")
            return result
            
        except Exception as e:
            # ✅ FIX: Return fallback instead of raising
            print(f"❌ Prompt optimization error: {e}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # Return a minimal fallback
            return {
                'optimized_prompt': f"You are a medical AI assistant. Please respond to: {query}",
                'query_type': 'general_medical',
                'medical_specialty': 'general_medicine',
                'patient_info': PatientInfo(),
                'metrics': PromptMetrics(
                    length=len(query), 
                    token_estimate=len(query)//4, 
                    context_utilization=0.0, 
                    patient_specificity=0.0, 
                    medical_terminology_density=0.0
                ),
                'language': 'English',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _format_patient_info(self, patient_info: PatientInfo) -> str:
        """Format patient information for prompts"""
        info_parts = []
        if patient_info.name:
            info_parts.append(f"Name: {patient_info.name}")
        if patient_info.age:
            info_parts.append(f"Age: {patient_info.age}")
        if patient_info.gender:
            info_parts.append(f"Gender: {patient_info.gender}")
        if patient_info.mrn:
            info_parts.append(f"MRN: {patient_info.mrn}")
        if patient_info.chief_complaint:
            info_parts.append(f"Chief Complaint: {patient_info.chief_complaint}")
        
        return ' | '.join(info_parts) if info_parts else "Patient information not fully documented"
    
    def _calculate_metrics(self, prompt: str, context_docs: List[Dict], patient_info: PatientInfo) -> PromptMetrics:
        """Calculate prompt quality metrics"""
        length = len(prompt)
        token_estimate = length // 4
        context_utilization = min(len(context_docs) / self.max_context_docs, 1.0)
        
        # Patient specificity
        patient_fields = [patient_info.name, patient_info.age, patient_info.gender]
        patient_specificity = sum(1 for field in patient_fields if field) / len(patient_fields)
        
        # Medical terminology density
        medical_terms = 0
        for pattern in self.medical_patterns.values():
            medical_terms += len(re.findall(pattern, prompt, re.IGNORECASE))
        medical_terminology_density = min(medical_terms / 10, 1.0)
        
        return PromptMetrics(
            length=length,
            token_estimate=token_estimate,
            context_utilization=context_utilization,
            patient_specificity=patient_specificity,
            medical_terminology_density=medical_terminology_density
        )

# Test function
def test_english_optimizer():
    """Test the English-focused prompt optimizer"""
    print("🧪 Testing English Medical Prompt Optimizer...")
    
    optimizer = EnglishMedicalPromptOptimizer()
    
    test_query = "Patient Name: Rogers, Pamela\nAge: 45\nChief Complaint: Chest pain and shortness of breath"
    test_context = [
        {
            'text': 'Patient Rogers, Pamela, 45-year-old female presents with acute chest pain. Vital signs: BP 140/90, HR 95. ECG shows ST elevation.',
            'similarity': 0.85,
            'metadata': {'file_name': 'emergency_notes.pdf'}
        }
    ]
    
    result = optimizer.optimize_prompt(test_query, test_context)
    
    print(f"✅ Test complete:")
    print(f"   Query type: {result['query_type']}")
    print(f"   Specialty: {result['medical_specialty']}")
    print(f"   Language: {result['language']}")
    
    return True

if __name__ == "__main__":
    test_english_optimizer()
