import os
from typing import List, Dict, Optional
import PyPDF2
import docx
import logging
import re

class DocumentProcessor:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize document processor with better medical-aware processing"""
        self.logger = logging.getLogger(__name__)
        
        try:
            from sentence_transformers import SentenceTransformer
            device = 'cpu'
            self.embedding_model = SentenceTransformer(
                embedding_model, 
                device=device,
                trust_remote_code=False
            )
            self.embedding_model = self.embedding_model.to(device)
            self.logger.info(f"Document processor initialized with {embedding_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
                self.logger.info("✅ Fallback to all-MiniLM-L6-v2 successful")
            except:
                self.embedding_model = None
                self.logger.warning("❌ No embedding model available")

    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_extension = os.path.splitext(file_path).lower()
        print(f"🔍 Extracting from {file_extension} file")
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            print(f"❌ Text extraction failed: {e}")
            raise

    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF with proper resource management"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 100) -> List[str]:
        """Enhanced chunking that preserves medical context and structure"""
        if not text or not text.strip():
            print("❌ Empty text for chunking")
            return []
        
        # Clean and normalize text
        text = text.strip()
        
        # Try to preserve medical sections if they exist
        medical_section_patterns = [
            r'\n(?=(?:PATIENT|HISTORY|DIAGNOSIS|TREATMENT|MEDICATION|LAB|VITAL|ASSESSMENT|PLAN|CHIEF COMPLAINT|PHYSICAL EXAM))',
            r'\n(?=\d+\.\s)',  # Numbered sections
            r'\n(?=[A-Z][A-Z\s]+:)',  # ALL CAPS headers
            r'\n\n+'  # Paragraph breaks
        ]
        
        # Try to split by medical sections first
        sections = [text]
        for pattern in medical_section_patterns:
            new_sections = []
            for section in sections:
                new_sections.extend(re.split(pattern, section, flags=re.IGNORECASE))
            sections = [s for s in new_sections if s.strip()]
            if len(sections) > 1:
                break
        
        chunks = []
        for section in sections:
            words = section.split()
            
            if len(words) <= chunk_size:
                # Section fits in one chunk
                if section.strip():
                    chunks.append(section.strip())
            else:
                # Split large sections with overlap to preserve context
                for i in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    chunk_text = ' '.join(chunk_words).strip()
                    
                    if chunk_text:
                        chunks.append(chunk_text)
                    
                    # Break if we've processed all words
                    if i + chunk_size >= len(words):
                        break
        
        print(f"📦 Created {len(chunks)} context-aware chunks")
        return chunks

    def process_document(self, file_path: str, doc_type: str = "medical", 
                        language: str = "en", chunk_size: int = 400) -> List[Dict]:
        """Complete document processing with enhanced debugging"""
        try:
            print(f"\n🔍 Processing: {os.path.basename(file_path)}")
            print(f"📄 File size: {os.path.getsize(file_path)} bytes")
            
            # Extract text
            raw_text = self.extract_text_from_file(file_path)
            print(f"📝 Raw text extracted: {len(raw_text)} characters")
            
            if not raw_text or len(raw_text.strip()) < 10:
                print("❌ No meaningful text extracted!")
                return []
            
            # Clean text for medical processing
            cleaned_text = self._clean_medical_text(raw_text)
            print(f"🧹 Cleaned text: {len(cleaned_text)} characters")
            
            if not cleaned_text:
                print("❌ All text removed during cleaning!")
                return []
            
            # Create context-aware chunks
            chunks = self.chunk_text(cleaned_text, chunk_size=chunk_size)
            
            if not chunks:
                print("❌ No chunks created!")
                return []
            
            # Create processed chunks with enhanced metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "text": chunk,
                    "metadata": {
                        "file_name": os.path.basename(file_path),
                        "doc_type": doc_type,
                        "language": language,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk.split()),
                        "processing_method": "medical_aware_chunking",
                        "contains_patient_data": self._detect_patient_data(chunk)
                    }
                }
                processed_chunks.append(chunk_data)
            
            print(f"✅ Success: {len(chunks)} medical-aware chunks created\n")
            return processed_chunks
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise

    def _clean_medical_text(self, text: str) -> str:
        """Enhanced text cleaning for medical documents"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove OCR artifacts common in medical documents
        text = re.sub(r'[|\\]{2,}', '', text)
        text = re.sub(r'_{3,}', '', text)
        text = re.sub(r'-{3,}', ' ', text)
        text = re.sub(r'\.{3,}', '...', text)
        
        # Basic PHI removal (while preserving medical context)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE_REDACTED]', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', text)
        
        # Preserve medical abbreviations and measurements
        text = re.sub(r'\s+([mg|ml|kg|lb|cm|mm|mcg|IU|units?])\b', r' \1', text)
        
        return text.strip()

    def _detect_patient_data(self, text: str) -> bool:
        """Detect if chunk contains patient-specific information"""
        patient_indicators = [
            r'patient\s+name',
            r'patient\s*:',
            r'mr\.|mrs\.|ms\.|dr\.',
            r'\b[A-Z][a-z]+,\s*[A-Z][a-z]+\b',
            r'age\s*:\s*\d+',
            r'dob\s*:',
            r'medical\s+record',
            r'chart\s+#'
        ]
        
        for pattern in patient_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def get_embedding(self, text: str):
        """Get embedding with fallback"""
        if self.embedding_model is None:
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            return [float(int(hash_obj.hexdigest()[i:i+2], 16)) for i in range(0, 32, 2)]
        
        return self.embedding_model.encode(text)
