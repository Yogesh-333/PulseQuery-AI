from typing import Dict, List, Optional
import re
import logging

class PatientReportGenerator:
    def __init__(self):
        """Initialize patient report generator with medical templates"""
        self.logger = logging.getLogger(__name__)
        
    def extract_patient_name(self, query: str) -> Optional[str]:
        """Extract patient name from query"""
        # Handle formats like "Patient Name: Rogers, Pamela" or just "Rogers, Pamela"
        patterns = [
            r"Patient Name:\s*([^,]+,\s*[^,\n]+)",
            r"patient:\s*([^,]+,\s*[^,\n]+)",
            r"([A-Z][a-z]+,\s*[A-Z][a-z]+)",  # Last, First format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def create_patient_report_prompt(self, patient_name: str, context_docs: List[Dict]) -> str:
        """Create specialized prompt for comprehensive patient reporting"""
        
        # Build context from documents
        context_text = ""
        for i, doc in enumerate(context_docs[:5], 1):  # Limit to top 5 most relevant
            context_text += f"\n--- Document {i} ---\n{doc['text']}\n"
        
        prompt = f"""You are an expert medical AI assistant specializing in comprehensive patient reporting. 

TASK: Generate a detailed, structured medical report for patient "{patient_name}" based on the provided medical document excerpts.

REPORT STRUCTURE REQUIRED:
## PATIENT REPORT: {patient_name}

### 1. PATIENT IDENTIFICATION
- Full Name
- Demographics (age, gender, etc. if available)
- Medical Record identifiers (if mentioned)

### 2. CHIEF COMPLAINT & PRESENTING SYMPTOMS
- Primary reason for visit/consultation
- Current symptoms and duration

### 3. MEDICAL HISTORY
- Past medical conditions
- Previous hospitalizations
- Surgical history
- Family history (if relevant)

### 4. CURRENT DIAGNOSES
- Primary diagnosis
- Secondary diagnoses
- Differential diagnoses considered

### 5. TREATMENTS & INTERVENTIONS
- Current medications and dosages
- Procedures performed
- Treatment plans
- Therapeutic interventions

### 6. CLINICAL FINDINGS & RESULTS
- Physical examination findings
- Laboratory results
- Imaging studies
- Vital signs

### 7. RECOMMENDATIONS & FOLLOW-UP
- Next steps in treatment
- Follow-up appointments
- Patient education
- Monitoring requirements

MEDICAL DOCUMENTATION EXCERPTS:
{context_text}

INSTRUCTIONS:
- Create a professional, clinical-grade report
- Use medical terminology appropriately
- Include specific details from the provided excerpts
- If information is not available, note "Not documented in available records"
- Maintain patient confidentiality standards
- Structure the report with clear sections and bullet points

BEGIN COMPREHENSIVE MEDICAL REPORT:
"""
        
        return prompt
    
    def generate_patient_report(self, medgemma_instance, rag_system, patient_name: str) -> Dict:
        """Generate complete patient report using MedGemma and RAG"""
        try:
            # Search for patient-specific documents
            search_queries = [
                patient_name,
                f"patient {patient_name}",
                f"{patient_name} medical record",
                f"{patient_name} diagnosis treatment"
            ]
            
            all_context_docs = []
            for query in search_queries:
                docs = rag_system.search_relevant_context(query, max_docs=3)
                all_context_docs.extend(docs)
            
            # Remove duplicates and get top relevant documents
            seen_ids = set()
            unique_docs = []
            for doc in all_context_docs:
                if doc['id'] not in seen_ids:
                    unique_docs.append(doc)
                    seen_ids.add(doc['id'])
            
            # Sort by similarity and take top 5
            unique_docs = sorted(unique_docs, key=lambda x: x.get('similarity', 0), reverse=True)[:5]
            
            if not unique_docs:
                return {
                    "success": False,
                    "error": f"No medical records found for patient {patient_name}",
                    "patient_name": patient_name,
                    "suggestions": [
                        "Check if patient name is spelled correctly",
                        "Ensure medical records have been uploaded to the system",
                        "Try searching with partial name or different format"
                    ]
                }
            
            # Create specialized prompt
            report_prompt = self.create_patient_report_prompt(patient_name, unique_docs)
            
            # Generate report with higher token limit
            result = medgemma_instance.generate_text(
                prompt=report_prompt,
                max_tokens=800,  # Longer for comprehensive report
                temperature=0.3  # Lower temperature for more focused medical content
            )
            
            return {
                "success": True,
                "patient_name": patient_name,
                "report": result["generated_text"],
                "documents_used": len(unique_docs),
                "context_sources": [doc.get('metadata', {}).get('file_name', 'Unknown') for doc in unique_docs],
                "generation_info": {
                    "model": result.get("model", "MedGemma"),
                    "tokens_generated": result.get("output_tokens", 0),
                    "device": result.get("device", "Unknown")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Patient report generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "patient_name": patient_name
            }
    
    def format_report_for_display(self, report_data: Dict) -> str:
        """Format report for web display"""
        if not report_data["success"]:
            return f"""
            <div class="alert alert-danger">
                <h5>❌ Report Generation Failed</h5>
                <p><strong>Patient:</strong> {report_data["patient_name"]}</p>
                <p><strong>Error:</strong> {report_data["error"]}</p>
                {f'<p><strong>Suggestions:</strong></p><ul>{"".join([f"<li>{s}</li>" for s in report_data.get("suggestions", [])])}</ul>' if report_data.get("suggestions") else ""}
            </div>
            """
        
        report = report_data["report"]
        sources = ", ".join(report_data["context_sources"])
        
        return f"""
        <div class="card">
            <div class="card-header bg-primary text-white">
                <h5>📋 Medical Report: {report_data["patient_name"]}</h5>
            </div>
            <div class="card-body">
                <div class="medical-report" style="white-space: pre-wrap; font-family: monospace; line-height: 1.6;">
{report}
                </div>
                <hr>
                <div class="report-metadata">
                    <small class="text-muted">
                        <strong>Sources:</strong> {sources}<br>
                        <strong>Documents Used:</strong> {report_data["documents_used"]}<br>
                        <strong>Generated by:</strong> {report_data["generation_info"]["model"]} 
                        ({report_data["generation_info"]["tokens_generated"]} tokens)
                    </small>
                </div>
            </div>
        </div>
        """
