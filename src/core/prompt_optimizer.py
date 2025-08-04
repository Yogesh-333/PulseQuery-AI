# src/core/prompt_optimizer.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # We'll start with a basic stemmer
import spacy

# Ensure NLTK resources are downloaded (run once)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# Load spaCy model (ensure it's downloaded: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# --- Define Medical Stopwords (Initial Draft - can be expanded) ---
# These are words common in medical context that might not add semantic value for querying
# but are not in generic English stopwords.
MEDICAL_STOPWORDS = set([
    "patient", "patients", "disease", "condition", "treatment", "therapy",
    "syndrome", "disorder", "medical", "clinical", "hospital", "doctor",
    "nurse", "provider", "health", "care", "diagnosis", "prognosis",
    "medication", "drug", "medicine", "report", "history", "examination",
    "study", "case", "cases", "data", "results", "findings", "symptom", "symptoms"
])

# Combine with generic English stopwords
ALL_STOPWORDS = set(stopwords.words('english')).union(MEDICAL_STOPWORDS)

stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning: lowercasing, removing special characters and numbers.
    """
    text = text.lower() # Lowercasing
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def tokenize_text(text: str) -> list[str]:
    """
    Tokenizes text into words using NLTK's word_tokenize.
    """
    return nltk.word_tokenize(text)

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes stopwords (English + medical) from a list of tokens.
    """
    return [token for token in tokens if token not in ALL_STOPWORDS]

def stem_tokens(tokens: list[str]) -> list[str]:
    """
    Applies stemming to a list of tokens.
    (Note: For medical text, lemmatization with a domain-specific model often performs better
    than stemming. We can switch to spaCy's lemmatization later.)
    """
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text: str, apply_stemming: bool = False) -> list[str]:
    """
    Orchestrates the preprocessing steps: clean -> tokenize -> remove stopwords -> (optional) stem.
    Returns a list of preprocessed tokens.
    """
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    filtered_tokens = remove_stopwords(tokens)
    if apply_stemming:
        return stem_tokens(filtered_tokens)
    return filtered_tokens

# You can add a spaCy-based lemmatization function as an alternative/improvement later:
def lemmatize_text(text: str) -> list[str]:
    """
    Lemmatizes text using spaCy. More sophisticated than stemming.
    """
    doc = nlp(text)
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

# --- Testing the functions (optional, for quick verification) ---
if __name__ == "__main__":
    sample_medical_query = "Can a diabetic patient take ibuprofen daily for joint pain without complications, according to the hospital guidelines?"
    print(f"Original Query: {sample_medical_query}\n")

    # Using our custom preprocessing pipeline
    processed_tokens_stemmed = preprocess_text(sample_medical_query, apply_stemming=True)
    print(f"Processed Tokens (Stemmed): {processed_tokens_stemmed}")

    processed_tokens_no_stem = preprocess_text(sample_medical_query, apply_stemming=False)
    print(f"Processed Tokens (No Stemming): {processed_tokens_no_stem}\n")

    # Using spaCy's built-in lemmatization (often preferred for robustness)
    lemmatized_tokens = lemmatize_text(sample_medical_query)
    print(f"Lemmatized Tokens (spaCy): {lemmatized_tokens}")

    # Test with a generic query
    generic_query = "The quick brown fox jumps over the lazy dog."
    processed_generic = preprocess_text(generic_query)
    print(f"\nGeneric Query: {generic_query}")
    print(f"Processed Generic Tokens: {processed_generic}")


    # Continue from previous content in src/core/prompt_optimizer.py

from collections import defaultdict, Counter
import math

# --- N-gram Model Development ---

class NGramModel:
    def __init__(self, n_min: int = 1, n_max: int = 3):
        """
        Initializes the N-gram model.
        Args:
            n_min (int): Minimum N-gram size (e.g., 1 for unigrams).
            n_max (int): Maximum N-gram size (e.g., 3 for trigrams).
        """
        self.n_min = n_min
        self.n_max = n_max
        self.ngram_counts = {n: defaultdict(int) for n in range(n_min, n_max + 1)}
        self.context_counts = {n: defaultdict(int) for n in range(n_min, n_max)} # For denominators in probabilities
        self.vocab = set()
        self.total_tokens = 0
        self.lambda_weights = None # For interpolation, will be set after training

    def _get_ngrams(self, tokens: list[str], n: int):
        """Helper to generate n-grams from a list of tokens."""
        if n == 1:
            return tokens
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def train(self, corpus_tokens: list[list[str]]):
        """
        Trains the N-gram model from a list of tokenized sentences (corpus).
        Args:
            corpus_tokens (list[list[str]]): A list where each inner list is a tokenized sentence.
        """
        self.vocab = set(token for sentence in corpus_tokens for token in sentence)
        self.total_tokens = sum(len(s) for s in corpus_tokens)

        for sentence in corpus_tokens:
            # Add start/end tokens to handle sentence boundaries, especially for higher N-grams
            processed_sentence = ['<s>'] * (self.n_max - 1) + sentence + ['</s>']
            
            for n in range(self.n_min, self.n_max + 1):
                ngrams = self._get_ngrams(processed_sentence, n)
                for ngram in ngrams:
                    self.ngram_counts[n][ngram] += 1
                    
                    if n > 1:
                        # For context_counts, we count the (n-1)-gram context
                        context = ' '.join(ngram.split(' ')[:-1])
                        self.context_counts[n-1][context] += 1

        # Calculate interpolation weights (simple uniform distribution for now, can be optimized)
        # For Kneser-Ney or other advanced smoothing, weights would be dynamically calculated.
        # Here, we'll use a simple approach for interpolation: equal weights.
        self.lambda_weights = [1.0 / (self.n_max - self.n_min + 1)] * (self.n_max - self.n_min + 1)
        # If n_min is 1, lambda_weights[0] for unigram, lambda_weights[1] for bigram, etc.

    def _get_smoothed_prob(self, ngram_str: str, n: int, alpha: float = 0.001) -> float:
        """
        Calculates Laplace (add-alpha) smoothed probability for a single N-gram.
        Args:
            ngram_str (str): The N-gram string (e.g., "diabetic patient").
            n (int): The size of the N-gram.
            alpha (float): Smoothing parameter (add-alpha).
        """
        count_ngram = self.ngram_counts[n][ngram_str]
        
        if n == 1: # Unigram probability
            denominator = self.total_tokens
        else: # For bigrams and higher, denominator is count of (n-1)-gram context
            context_tokens = ' '.join(ngram_str.split(' ')[:-1])
            denominator = self.context_counts[n-1][context_tokens]
            
        # Add-alpha smoothing
        numerator = count_ngram + alpha
        denominator += alpha * len(self.vocab) # For unigrams, it's vocab size; for higher, it's (context_count + alpha * vocab_size) if context doesn't exist
        
        # If context is not found (happens for very sparse data), fall back to unigram or a default small prob
        if denominator == 0:
            return alpha / len(self.vocab) # Minimal probability if context unseen
        
        return numerator / denominator

    def calculate_interpolated_prob(self, target_ngram_str: str, n: int) -> float:
        """
        Calculates the interpolated probability for a target N-gram.
        Uses a linear interpolation of (n)-gram, (n-1)-gram, ..., (n_min)-gram probabilities.
        """
        if self.lambda_weights is None:
            raise ValueError("Model not trained. Call train() first.")

        prob = 0.0
        # Iterate from target N-gram size down to n_min
        for current_n in range(n, self.n_min - 1, -1):
            if current_n == 0: # Base case for single token (unigram or fallback)
                prob += self.lambda_weights[0] * (1.0 / len(self.vocab)) # Use 0-gram (uniform distribution)
                break

            if current_n == 1: # Unigram
                target_token = target_ngram_str.split(' ')[-1] # Last token is the unigram
                prob_current_n = self._get_smoothed_prob(target_token, 1)
                prob += self.lambda_weights[current_n - self.n_min] * prob_current_n
            else: # Bigram, Trigram, etc.
                if len(target_ngram_str.split(' ')) < current_n:
                    continue # Skip if target ngram is smaller than current_n

                # Extract the current_n-gram from the end of the target_ngram_str
                # e.g., if target="a b c d" and current_n=3, we want "b c d"
                current_ngram_part = ' '.join(target_ngram_str.split(' ')[-current_n:])
                
                prob_current_n = self._get_smoothed_prob(current_ngram_part, current_n)
                
                # Check if this ngram contributes to the interpolation (it should be in the range [n_min, n_max])
                if current_n >= self.n_min:
                    # The index for lambda_weights corresponds to the order of n-gram sizes.
                    # If n_min=1, n_max=3, then lambda_weights[0] is for unigram, [1] for bigram, [2] for trigram.
                    lambda_idx = current_n - self.n_min
                    if lambda_idx < len(self.lambda_weights):
                        prob += self.lambda_weights[lambda_idx] * prob_current_n
                    else:
                        # This scenario indicates an issue with lambda_weights setup if we're interpolating up to n.
                        # For now, just continue, but it might need refinement based on actual interpolation formula.
                        pass
        return prob
    
    def score_sentence_fluency(self, tokens: list[str]) -> float:
        """
        Calculates the log probability (fluency score) of a sentence using the trained interpolated N-gram model.
        Args:
            tokens (list[str]): A list of preprocessed tokens representing the sentence.
        Returns:
            float: The log probability of the sentence. Higher score means more fluent/probable.
        """
        if not self.ngram_counts[self.n_min]:
            raise ValueError("N-gram model not trained. Call train() first.")

        # Add start/end tokens consistent with training
        processed_tokens = ['<s>'] * (self.n_max - 1) + tokens + ['</s>']
        
        log_prob_sum = 0.0
        
        for i in range(self.n_max - 1, len(processed_tokens)): # Start from where actual words begin
            # For each word, consider its context up to n_max
            target_token = processed_tokens[i]
            
            # Form the N-gram ending with target_token, trying from n_max down to n_min
            # e.g., if n_max=3, for current word 'd', consider 'b c d' then 'c d' then 'd'
            
            # The context should be (n-1) tokens before the target_token
            # We are calculating P(word_i | word_i-1, word_i-2 ...)
            
            # Here's a simplification for interpolation:
            # We calculate P_interpolated(word_i | context_i-1)
            # This is typically sum(lambda_n * P(word_i | (n-1)-gram context))
            
            # Let's adjust the scoring to calculate P(token_i | history)
            # We'll use the interpolated probability for the (n_max)-gram ending with token_i
            
            # Create the current N-gram string that ends with the current token
            if i >= self.n_max - 1:
                current_ngram_str = ' '.join(processed_tokens[i - (self.n_max - 1) : i + 1])
                prob = self.calculate_interpolated_prob(current_ngram_str, self.n_max)
                log_prob_sum += math.log(prob) if prob > 0 else -float('inf')
            
        return log_prob_sum

# --- Testing the N-gram model (add to your if __name__ == "__main__": block) ---
if __name__ == "__main__":
    # ... (previous test code for preprocessing) ...

    # --- Test N-gram Model ---
    print("\n--- Testing N-gram Model ---")

    # Sample medical corpus (for demonstration)
    # In a real scenario, this would be loaded from your data/processed/ directory
    # and would be much larger.
    sample_corpus = [
        "A patient presented with severe abdominal pain and fever.",
        "The patient had a history of diabetes and hypertension.",
        "Management included antibiotics and pain relief.",
        "Fever persisted despite medication.",
        "The patient with diabetes required careful monitoring."
    ]

    # Preprocess the sample corpus
    tokenized_corpus = [preprocess_text(sentence, apply_stemming=False) for sentence in sample_corpus]
    print(f"Tokenized Corpus (first 2 sentences):\n{tokenized_corpus[:2]}\n")

    # Initialize and train the N-gram model
    ngram_model = NGramModel(n_min=1, n_max=3)
    ngram_model.train(tokenized_corpus)
    print("N-gram model trained.")
    # print(f"Unigram counts (sample): {list(ngram_model.ngram_counts[1].items())[:5]}")
    # print(f"Bigram counts (sample): {list(ngram_model.ngram_counts[2].items())[:5]}")
    # print(f"Trigram counts (sample): {list(ngram_model.ngram_counts[3].items())[:5]}")


    # Test sentence scoring
    test_sentence_good = "The patient had history diabetes." # "The patient had a history of diabetes"
    test_sentence_bad = "Cat quickly sky green."

    # Preprocess test sentences
    processed_good = preprocess_text(test_sentence_good, apply_stemming=False)
    processed_bad = preprocess_text(test_sentence_bad, apply_stemming=False)

    print(f"\nScoring sentence: '{test_sentence_good}' -> {processed_good}")
    score_good = ngram_model.score_sentence_fluency(processed_good)
    print(f"Fluency Score: {score_good:.4f}")

    print(f"\nScoring sentence: '{test_sentence_bad}' -> {processed_bad}")
    score_bad = ngram_model.score_sentence_fluency(processed_bad)
    print(f"Fluency Score: {score_bad:.4f}")

    # Example: A query that should score higher than random words
    medical_query_tokens = preprocess_text("Can diabetic patient take ibuprofen daily for joint pain", apply_stemming=False)
    query_score = ngram_model.score_sentence_fluency(medical_query_tokens)
    print(f"\nScoring query: 'Can diabetic patient take ibuprofen daily for joint pain' -> {medical_query_tokens}")
    print(f"Fluency Score: {query_score:.4f}")

    # Continue from previous content in src/core/prompt_optimizer.py

from sentence_transformers import SentenceTransformer, util
import numpy as np

# --- Semantic Similarity Module ---

class SemanticSimilarityModel:
    def __init__(self, model_name: str = 'emilyalsentzer/Bio_ClinicalBERT'):
        """
        Initializes the semantic similarity model.
        Args:
            model_name (str): Name of the pre-trained Sentence Transformer model to use.
                              'emilyalsentzer/Bio_ClinicalBERT' is a good medical-domain choice.
                              'all-MiniLM-L6-v2' is a general smaller model for quick testing if BioBERT is slow.
        """
        print(f"Loading Sentence Transformer model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            print("Please ensure you have an internet connection or the model is cached locally.")
            # Fallback to a smaller, more common model if primary fails
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Falling back to 'all-MiniLM-L6-v2'.")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates a sentence embedding for the given text.
        Args:
            text (str): The input text.
        Returns:
            np.ndarray: The embedding vector.
        """
        return self.model.encode(text, convert_to_numpy=True)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the cosine similarity between two texts.
        Args:
            text1 (str): First input text.
            text2 (str): Second input text.
        Returns:
            float: Cosine similarity score (between -1 and 1).
        """
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        # util.cos_sim returns a tensor, convert to float
        return util.cos_sim(embedding1, embedding2).item()


# --- Testing the functions (add to your if __name__ == "__main__": block) ---
if __name__ == "__main__":
    # ... (previous test code for preprocessing and N-gram model) ...

    # --- Test Semantic Similarity Model ---
    print("\n--- Testing Semantic Similarity Model ---")

    # Initialize semantic model (can be slow on first run as it downloads)
    # Using 'all-MiniLM-L6-v2' for faster testing, switch to BioBERT later.
    # semantic_model = SemanticSimilarityModel(model_name='emilyalsentzer/Bio_ClinicalBERT')
    semantic_model = SemanticSimilarityModel(model_name='all-MiniLM-L6-v2') # Faster for initial tests

    original_query = "Can a diabetic patient take ibuprofen daily for joint pain?"
    paraphrased_query_good = "Is it safe for a person with diabetes to take ibuprofen every day for their joint aches?"
    paraphrased_query_bad = "Can a healthy person eat a lot of candy?"
    unrelated_text = "The quick brown fox jumps over the lazy dog."

    similarity_good = semantic_model.calculate_similarity(original_query, paraphrased_query_good)
    similarity_bad = semantic_model.calculate_similarity(original_query, paraphrased_query_bad)
    similarity_unrelated = semantic_model.calculate_similarity(original_query, unrelated_text)

    print(f"Original Query: \"{original_query}\"")
    print(f"Good Paraphrase: \"{paraphrased_query_good}\"")
    print(f"Similarity (Good): {similarity_good:.4f}")

    print(f"\nBad Paraphrase: \"{paraphrased_query_bad}\"")
    print(f"Similarity (Bad): {similarity_bad:.4f}")

    print(f"\nUnrelated Text: \"{unrelated_text}\"")
    print(f"Similarity (Unrelated): {similarity_unrelated:.4f}")


# Continue from previous content in src/core/prompt_optimizer.py

# --- Prompt Optimization and Scoring ---

def generate_candidate_prompts(original_query: str) -> list[str]:
    """
    Generates a list of candidate optimized prompts from an original query.
    For this initial phase, this is a placeholder with hardcoded examples
    or simple rule-based transformations.
    In later stages, this could involve MedGemma for paraphrasing.
    """
    candidates = [
        original_query, # Include the original query as a candidate
        "Is it safe for a diabetic to take ibuprofen daily for joint pain?",
        "Daily ibuprofen use in diabetic patients for arthralgia: safety concerns?",
        "Recommend safe dosage of ibuprofen for diabetic joint pain.",
        "Management of joint pain in diabetes with daily ibuprofen.",
        "Ibuprofen use for diabetic patients' joint pain guidelines.",
        "What are the implications of daily ibuprofen use for diabetics with joint pain?"
    ]
    # You could add simple transformations here:
    # candidates.append(original_query.replace("patient", "individual"))
    return candidates

def score_candidate_prompt(
    original_query: str,
    candidate_prompt: str,
    ngram_model: NGramModel,
    semantic_model: SemanticSimilarityModel,
    fluency_weight: float = 0.3,    # Weight for N-gram fluency
    similarity_weight: float = 0.7  # Weight for semantic similarity
) -> dict:
    """
    Calculates a composite score for a candidate prompt.
    Combines N-gram fluency and semantic similarity.
    Returns a dictionary with individual scores and the composite score.
    """
    # 1. Fluency Score (N-gram)
    # Ensure tokens are consistent with how the N-gram model was trained (e.g., no stemming)
    candidate_tokens_for_fluency = preprocess_text(candidate_prompt, apply_stemming=False)
    # Handle empty tokens list if preprocessing results in no meaningful tokens
    if not candidate_tokens_for_fluency:
        fluency_score = -float('inf') # Assign a very low score
    else:
        fluency_score = ngram_model.score_sentence_fluency(candidate_tokens_for_fluency)

    # 2. Semantic Similarity Score (Cosine Similarity)
    similarity_score = semantic_model.calculate_similarity(original_query, candidate_prompt)

    # Normalize scores (optional but good for combining disparate metrics)
    # Fluency scores are log probabilities (negative). We want higher for 'better'.
    # A simple normalization for log_prob: map to a 0-1 range or just use directly.
    # For now, we'll use log_prob directly for weighting, understanding it's negative.
    # Higher fluency_score (closer to 0) is better.
    # Higher similarity_score (closer to 1) is better.

    # Composite score: A simple weighted sum.
    # Adjusting fluency_score to be positive or relative for intuitive weighting
    # Example: If max_possible_log_prob is 0 (for a perfect match), then (0 - fluency_score) is an 'error' score.
    # Or, we can just acknowledge fluency_score is negative and higher (less negative) is better.
    # Let's use it as is, aiming for a "higher is better" composite score.
    # So, fluency_weight would be applied to its absolute value or after scaling.
    # For simplicity, let's consider relative scores:
    # Fluency is often negative. Higher means 'less negative', closer to 0.
    # Similarity is 0 to 1. Higher means 'better'.

    # For a composite score where 'higher is better':
    # We can use similarity directly (0 to 1).
    # For fluency, let's normalize it relative to a baseline or make it positive.
    # A simple way to make fluency "positive" and relatively comparable is to take e^score, but this can be very small.
    # Or simply: composite = (similarity * similarity_weight) + (scaled_fluency * fluency_weight)
    # Let's just use raw scores for now and ensure weights are set appropriately.
    composite_score = (similarity_score * similarity_weight) + (fluency_score * fluency_weight)

    return {
        "fluency_score": fluency_score,
        "similarity_score": similarity_score,
        "composite_score": composite_score
    }

# --- Testing the functions (add to your if __name__ == "__main__": block) ---
if __name__ == "__main__":
    # ... (previous test code for preprocessing, N-gram model, semantic similarity model) ...

    # --- Test Prompt Optimization and Scoring ---
    print("\n--- Testing Prompt Optimization and Scoring ---")

    # Assuming ngram_model and semantic_model are already initialized from previous tests
    # You might want to move their initialization outside the if __name__ block
    # or ensure they are only initialized once if running the full script.

    # Example: Initialize only if not already done (for isolated execution)
    if 'ngram_model' not in locals():
        # Re-initialize N-gram model and train it for this section if needed
        sample_corpus = [
            "A patient presented with severe abdominal pain and fever.",
            "The patient had a history of diabetes and hypertension.",
            "Management included antibiotics and pain relief.",
            "Fever persisted despite medication.",
            "The patient with diabetes required careful monitoring."
        ]
        tokenized_corpus = [preprocess_text(sentence, apply_stemming=False) for sentence in sample_corpus]
        ngram_model = NGramModel(n_min=1, n_max=3)
        ngram_model.train(tokenized_corpus)

    if 'semantic_model' not in locals():
        semantic_model = SemanticSimilarityModel(model_name='all-MiniLM-L6-v2') # Or 'emilyalsentzer/Bio_ClinicalBERT'


    original_user_query = "Can a diabetic patient take ibuprofen daily for joint pain without complications?"
    print(f"Original User Query: \"{original_user_query}\"\n")

    candidate_prompts = generate_candidate_prompts(original_user_query)

    scored_candidates = []
    for i, candidate in enumerate(candidate_prompts):
        scores = score_candidate_prompt(
            original_user_query,
            candidate,
            ngram_model,
            semantic_model,
            fluency_weight=0.3,
            similarity_weight=0.7 # These weights can be tuned
        )
        scored_candidates.append({
            "id": i,
            "prompt": candidate,
            "scores": scores
        })
        print(f"Candidate {i+1}: \"{candidate}\"")
        print(f"  Fluency: {scores['fluency_score']:.4f}")
        print(f"  Similarity: {scores['similarity_score']:.4f}")
        print(f"  Composite: {scores['composite_score']:.4f}\n")

    # Rank candidates by composite score (highest is best)
    ranked_candidates = sorted(scored_candidates, key=lambda x: x['scores']['composite_score'], reverse=True)

    print("--- Ranked Optimized Prompts ---")
    for rank, item in enumerate(ranked_candidates):
        print(f"Rank {rank+1} (Score: {item['scores']['composite_score']:.4f}): \"{item['prompt']}\"")

    # Continue from previous content in src/core/prompt_optimizer.py

# Import the new SessionManager at the top of prompt_optimizer.py
from src.services.session_manager import SessionManager

# ... (rest of your existing code) ...

# --- Testing the functions (updated if __name__ == "__main__": block) ---
if __name__ == "__main__":
    # ... (all your existing test code for preprocessing, N-gram, semantic, and prompt optimization) ...

    # --- Test Session Management ---
    print("\n--- Testing Session Management ---")

    session_manager = SessionManager() # Get the singleton instance

    # Simulate a user ID (e.g., from a login)
    test_user_id = "clinician_john_doe"

    # Capture the results from your prompt optimization test
    # (assuming `original_user_query` and `ranked_candidates` are available from previous test)

    # Store the results of the optimization as a session
    session_data_1 = {
        "original_query": original_user_query,
        "optimized_prompts": [
            {"prompt": item['prompt'], "composite_score": item['scores']['composite_score']}
            for item in ranked_candidates
        ],
        "selected_prompt": ranked_candidates[0]['prompt'] if ranked_candidates else None,
        "feedback": "N/A" # Placeholder for future user feedback
    }
    session_id_1 = session_manager.add_session_data(test_user_id, session_data_1)

    # Simulate another query and session
    original_user_query_2 = "What are the latest guidelines for treating type 2 diabetes?"
    candidate_prompts_2 = generate_candidate_prompts(original_user_query_2) # Using the same dummy generator for now

    # Example: score only the original query as "optimized" for simplicity in this test
    # In reality, you'd run full optimization for this query too
    scored_original_2 = score_candidate_prompt(
        original_user_query_2,
        original_user_query_2,
        ngram_model,
        semantic_model,
        fluency_weight=0.3,
        similarity_weight=0.7
    )
    session_data_2 = {
        "original_query": original_user_query_2,
        "optimized_prompts": [{"prompt": original_user_query_2, "composite_score": scored_original_2['composite_score']}],
        "selected_prompt": original_user_query_2,
        "feedback": "Good enough"
    }
    session_id_2 = session_manager.add_session_data(test_user_id, session_data_2)


    print(f"\n--- Session History for '{test_user_id}' ---")
    history = session_manager.get_session_history(test_user_id)
    for session in history:
        print(f"Session ID: {session_manager._sessions[test_user_id].get(list(session_manager._sessions[test_user_id].keys())[list(session_manager._sessions[test_user_id].values()).index(session)], 'N/A')}")
        print(f"  Timestamp: {session['timestamp']}")
        print(f"  Original Query: \"{session['original_query']}\"")
        print(f"  Selected Prompt: \"{session['selected_prompt']}\"")
        if session['optimized_prompts']:
            print(f"  Top Optimized Score: {session['optimized_prompts'][0]['composite_score']:.4f}")
        print("-" * 20)

    # You can also retrieve a specific session
    retrieved_session = session_manager.get_session_by_id(test_user_id, session_id_1)
    if retrieved_session:
        print(f"\n--- Retrieved Session {session_id_1} ---")
        print(f"Original Query: \"{retrieved_session['original_query']}\"")
        print(f"Selected Prompt: \"{retrieved_session['selected_prompt']}\"")

from src.database.chromadb_manager import ChromaDBManager
if __name__ == "__main__":
    # ... (all your existing test code for preprocessing, N-gram, semantic, prompt optimization, and session management) ...

    # --- Test ChromaDBManager (RAG Foundation) ---
    print("\n--- Testing ChromaDBManager (RAG Foundation) ---")

    # Important: Set persist_directory to a path relative to your project root
    # or an absolute path where you want the db files to be stored.
    # Make sure this directory exists or can be created.
    chroma_db_dir = "./chroma_data"
    
    # Initialize ChromaDB Manager
    # Use the same embedding model as your semantic similarity for consistency
    db_manager = ChromaDBManager(
        persist_directory=chroma_db_dir,
        embedding_model_name='all-MiniLM-L6-v2' # Must match semantic_model
    )

    # Optional: Clear collection if you want to start fresh every run
    # db_manager.delete_collection()
    # db_manager = ChromaDBManager(persist_directory=chroma_db_dir, embedding_model_name='all-MiniLM-L6-v2')

    # Add some sample medical documents (chunks from your corpus/guidelines)
    sample_medical_docs = [
        {"text": "Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) used for pain relief and fever reduction.", "metadata": {"source": "Drug_Info", "section": "Ibuprofen_Overview"}},
        {"text": "Diabetic patients should use NSAIDs with caution due to potential renal impairment and cardiovascular risks.", "metadata": {"source": "Diabetes_Guidelines", "section": "NSAID_Use"}},
        {"text": "For joint pain in diabetics, acetaminophen is often preferred as a first-line agent, if not contraindicated.", "metadata": {"source": "Diabetes_Guidelines", "section": "Pain_Management"}},
        {"text": "Regular monitoring of kidney function is recommended when NSAIDs are prescribed to diabetic patients.", "metadata": {"source": "Monitoring_Protocols", "section": "Renal_Function"}},
        {"text": "Type 2 diabetes mellitus management includes diet, exercise, and oral hypoglycemic agents or insulin.", "metadata": {"source": "Diabetes_Overview", "section": "Type2_Treatment"}}
    ]

    # Only add documents if the collection is empty, to avoid duplicates on every run
    if db_manager.get_document_count() == 0:
        db_manager.add_documents(sample_medical_docs)
    else:
        print(f"ChromaDB collection already contains {db_manager.get_document_count()} documents. Skipping re-adding.")

    # Test querying
    query_for_rag = "Can a diabetic patient take ibuprofen for joint pain?"
    print(f"\nQuerying ChromaDB for: \"{query_for_rag}\"")
    retrieved_docs = db_manager.query_documents(query_texts=[query_for_rag], n_results=2)

    if retrieved_docs and retrieved_docs[0]['results']:
        print("\n--- Retrieved Documents ---")
        for doc_info in retrieved_docs[0]['results']:
            print(f"  Document: \"{doc_info['document']}\"")
            print(f"  Source: {doc_info['metadata'].get('source', 'N/A')}")
            print(f"  Distance (lower is better): {doc_info['distance']:.4f}\n")
    else:
        print("No documents retrieved.")

    # Another query example
    query_for_rag_2 = "Latest treatments for type 2 diabetes"
    print(f"\nQuerying ChromaDB for: \"{query_for_rag_2}\"")
    retrieved_docs_2 = db_manager.query_documents(query_texts=[query_for_rag_2], n_results=1)
    if retrieved_docs_2 and retrieved_docs_2[0]['results']:
        print("\n--- Retrieved Documents (Query 2) ---")
        for doc_info in retrieved_docs_2[0]['results']:
            print(f"  Document: \"{doc_info['document']}\"")
            print(f"  Source: {doc_info['metadata'].get('source', 'N/A')}")
            print(f"  Distance: {doc_info['distance']:.4f}\n")
    else:
        print("No documents retrieved for query 2.")

from src.core.medgemma_inference import MedGemmaInference
if __name__ == "__main__":
    # ... (all your existing test code for preprocessing, N-gram, semantic, prompt optimization, session management, and ChromaDBManager) ...

    # --- Test MedGemma Inference (with RAG context) ---
    print("\n--- Testing MedGemma Inference (with RAG context) ---")

    # Initialize MedGemma Inference (placeholder)
    medgemma_inference = MedGemmaInference()

    # Reuse optimized prompt from previous test
    best_optimized_prompt_1 = ranked_candidates[0]['prompt'] if ranked_candidates else original_user_query
    print(f"Using optimized prompt: \"{best_optimized_prompt_1}\"")

    # Reuse retrieved documents from ChromaDB query 1
    retrieved_context_1 = retrieved_docs[0]['results'] if retrieved_docs and retrieved_docs[0]['results'] else []
    
    # Generate answer for Query 1
    generated_answer_1 = medgemma_inference.generate_answer(best_optimized_prompt_1, retrieved_context_1)
    print(f"\nGenerated Answer (Query 1):")
    print(generated_answer_1)
    print("-" * 50)

    # Reuse optimized prompt from previous test (Query 2)
    original_user_query_2 = "What are the latest guidelines for treating type 2 diabetes?"
    # For simplicity, we'll use the original query as optimized for this test run
    best_optimized_prompt_2 = original_user_query_2 

    # Reuse retrieved documents from ChromaDB query 2
    retrieved_context_2 = retrieved_docs_2[0]['results'] if retrieved_docs_2 and retrieved_docs_2[0]['results'] else []

    # Generate answer for Query 2
    generated_answer_2 = medgemma_inference.generate_answer(best_optimized_prompt_2, retrieved_context_2)
    print(f"\nGenerated Answer (Query 2):")
    print(generated_answer_2)
    print("-" * 50)