import spacy
from spacy.matcher import PhraseMatcher

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Initialize the matcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# Sample HCC conditions dictionary (In production, this loads from your MSSQL crosswalk)
hcc_conditions = ["type 2 diabetes", "major depressive disorder", "congestive heart failure", "copd", "rheumatoid arthritis"]
patterns = [nlp.make_doc(text) for text in hcc_conditions]
matcher.add("CHRONIC_CONDITIONS", patterns)

# Words that invalidate a diagnosis for active HCC coding
exclusion_modifiers = ["history of", "family history of", "resolved", "ruled out", "negative for", "denies"]

def extract_conditions(clinical_note: str):
    """
    Parses a clinical note, extracts chronic conditions, and flags 
    if they are historical/negated based on surrounding text context.
    """
    doc = nlp(clinical_note)
    matches = matcher(doc)
    
    extracted_data = []
    
    for match_id, start, end in matches:
        condition_span = doc[start:end]
        
        # Look at the 5 words immediately preceding the condition for context
        preceding_context = doc[max(0, start - 5):start].text.lower()
        
        # Check if any exclusion modifiers are in the preceding context
        is_historical_or_negated = any(modifier in preceding_context for modifier in exclusion_modifiers)
        
        extracted_data.append({
            "condition": condition_span.text,
            "is_active": not is_historical_or_negated,
            "context_snippet": doc[max(0, start - 5):min(len(doc), end + 5)].text
        })
        
    return extracted_data

# --- Test the Pipeline ---
if __name__ == "__main__":
    sample_chart = """
    Patient presents for a routine follow-up. 
    Patient denies any chest pain. 
    Family history of Congestive Heart Failure. 
    Patient is currently managing Type 2 Diabetes with Metformin.
    """
    
    results = extract_conditions(sample_chart)
    
    print("--- NLP Extraction Results ---")
    for res in results:
        status = "Active" if res["is_active"] else "Invalid (Historical/Negated)"
        print(f"Condition Found: {res['condition']}")
        print(f"Coding Status: {status}")
        print(f"Audit Snippet: '{res['context_snippet']}'\n")