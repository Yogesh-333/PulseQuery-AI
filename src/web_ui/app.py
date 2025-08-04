# src/web_ui/app.py

import streamlit as st
from src.core.prompt_optimizer import preprocess_text, NGramModel, SemanticSimilarityModel, generate_candidate_prompts, score_candidate_prompt
from src.database.chromadb_manager import ChromaDBManager
from src.core.medgemma_inference import MedGemmaInference
from src.services.session_manager import SessionManager # Assuming this is available

# --- Configuration ---
# These should ideally come from config files (e.g., config/app_config.yaml)
# For now, hardcode them for quick setup.
CHROMA_DB_DIR = "./chroma_data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Match what you used for ChromaDB and SemanticSimilarity
FLUENCY_WEIGHT = 0.3
SIMILARITY_WEIGHT = 0.7

# --- Initialize Global Models and Managers (Singleton/Cached) ---
@st.cache_resource # Cache resource to load models only once across user sessions
def get_ngram_model():
    # Re-initialize N-gram model and train it
    # This corpus should ideally be loaded from a persistent source in data/processed
    # For demo, using the same small corpus as in prompt_optimizer.py
    sample_corpus = [
        "A patient presented with severe abdominal pain and fever.",
        "The patient had a history of diabetes and hypertension.",
        "Management included antibiotics and pain relief.",
        "Fever persisted despite medication.",
        "The patient with diabetes required careful monitoring."
    ]
    tokenized_corpus = [preprocess_text(sentence, apply_stemming=False) for sentence in sample_corpus]
    model = NGramModel(n_min=1, n_max=3)
    model.train(tokenized_corpus)
    return model

@st.cache_resource
def get_semantic_model():
    return SemanticSimilarityModel(model_name=EMBEDDING_MODEL_NAME)

@st.cache_resource
def get_db_manager():
    manager = ChromaDBManager(
        persist_directory=CHROMA_DB_DIR,
        collection_name="medical_knowledge_base",
        embedding_model_name=EMBEDDING_MODEL_NAME
    )
    # Add sample docs if db is empty for demonstration purposes
    sample_medical_docs = [
        {"text": "Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) used for pain relief and fever reduction.", "metadata": {"source": "Drug_Info", "section": "Ibuprofen_Overview"}},
        {"text": "Diabetic patients should use NSAIDs with caution due to potential renal impairment and cardiovascular risks.", "metadata": {"source": "Diabetes_Guidelines", "section": "NSAID_Use"}},
        {"text": "For joint pain in diabetics, acetaminophen is often preferred as a first-line agent, if not contraindicated.", "metadata": {"source": "Diabetes_Guidelines", "section": "Pain_Management"}},
        {"text": "Regular monitoring of kidney function is recommended when NSAIDs are prescribed to diabetic patients.", "metadata": {"source": "Monitoring_Protocols", "section": "Renal_Function"}},
        {"text": "Type 2 diabetes mellitus management includes diet, exercise, and oral hypoglycemic agents or insulin.", "metadata": {"source": "Diabetes_Overview", "section": "Type2_Treatment"}}
    ]
    if manager.get_document_count() == 0:
        manager.add_documents(sample_medical_docs)
    return manager

@st.cache_resource
def get_medgemma_inference():
    return MedGemmaInference() # Placeholder model

# Session Manager is a singleton, so no need for st.cache_resource for its instance
session_manager = SessionManager()

# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="PulseQuery AI")

st.title("🩺 PulseQuery AI: Intelligent Medical Q&A")
st.markdown("Your secure, AI-powered assistant for evidence-backed medical information.")

# User ID input (for session management)
user_id = st.sidebar.text_input("Enter your User ID (e.g., clinician_001)", value="demo_user")
if not user_id:
    st.sidebar.warning("Please enter a User ID.")
    st.stop() # Stop execution if no user ID

# Main query input
original_query = st.text_area("Enter your medical question here:", height=100,
                              placeholder="e.g., Can a diabetic patient take ibuprofen daily for joint pain without complications?")

# Action Button
if st.button("Get AI-Powered Answer"):
    if not original_query.strip():
        st.warning("Please enter a query to get an answer.")
    else:
        st.info("Processing your query...")
        
        # Initialize models (will be cached after first run)
        ngram_model = get_ngram_model()
        semantic_model = get_semantic_model()
        db_manager = get_db_manager()
        medgemma_inference = get_medgemma_inference()

        with st.spinner("Optimizing prompt, retrieving context, and generating answer..."):
            # 1. Prompt Optimization
            candidate_prompts = generate_candidate_prompts(original_query)
            scored_candidates = []
            for candidate in candidate_prompts:
                scores = score_candidate_prompt(
                    original_query,
                    candidate,
                    ngram_model,
                    semantic_model,
                    fluency_weight=FLUENCY_WEIGHT,
                    similarity_weight=SIMILARITY_WEIGHT
                )
                scored_candidates.append({
                    "prompt": candidate,
                    "scores": scores
                })
            ranked_candidates = sorted(scored_candidates, key=lambda x: x['scores']['composite_score'], reverse=True)
            best_optimized_prompt = ranked_candidates[0]['prompt'] if ranked_candidates else original_query

            # 2. RAG - Retrieve Context
            st.subheader("Retrieved Context from Local Knowledge Base:")
            retrieved_docs_rag = db_manager.query_documents(query_texts=[best_optimized_prompt], n_results=3)
            context_for_llm = []
            if retrieved_docs_rag and retrieved_docs_rag[0]['results']:
                for i, doc_info in enumerate(retrieved_docs_rag[0]['results']):
                    st.write(f"- **Source**: {doc_info['metadata'].get('source', 'N/A')}")
                    st.text(f"  {doc_info['document']}")
                    context_for_llm.append(doc_info) # Store for MedGemma

            # 3. MedGemma Inference
            st.subheader("AI-Generated Answer:")
            generated_answer = medgemma_inference.generate_answer(best_optimized_prompt, context_for_llm)
            st.write(generated_answer)

            # 4. Store Session Data
            session_data = {
                "original_query": original_query,
                "optimized_prompts_ranked": [
                    {"prompt": item['prompt'], "composite_score": item['scores']['composite_score']}
                    for item in ranked_candidates
                ],
                "best_optimized_prompt": best_optimized_prompt,
                "retrieved_context": context_for_llm,
                "generated_answer": generated_answer,
                "user_feedback": "N/A" # Placeholder for future feedback mechanism
            }
            session_manager.add_session_data(user_id, session_data)

            st.success("Processing Complete!")

# --- Display Session History (Sidebar) ---
st.sidebar.subheader("Session History")
history_data = session_manager.get_session_history(user_id)
if history_data:
    for i, session in enumerate(history_data):
        with st.sidebar.expander(f"Query {i+1}: {session['original_query'][:50]}..."):
            st.write(f"**Timestamp**: {session['timestamp']}")
            st.write(f"**Original Query**: {session['original_query']}")
            st.write(f"**Best Optimized Prompt**: {session['best_optimized_prompt']}")
            st.write(f"**AI Answer**: {session['generated_answer'][:100]}...")
            # Optionally show more details like full optimized_prompts_ranked, retrieved_context
else:
    st.sidebar.info("No history yet.")