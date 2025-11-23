import os
from typing import Dict, List, Any
from groq import Groq
from transformers import pipeline

def _extract_keywords_with_legalbert(user_query: str) -> List[str]:
    """
    Uses LegalBERT to extract key legal terms from a query.
    """
    try:
        # Load a pre-trained LegalBERT model for Named Entity Recognition (NER)
        nlp = pipeline("ner", model="Akshita/legal-ner", aggregation_strategy="simple")
        entities = nlp(user_query)
        # Extract unique terms while preserving order
        extracted_terms = list(dict.fromkeys([entity['word'] for entity in entities]))
        return extracted_terms
    except Exception as e:
        # If there's an error (e.g., model download fails), return an empty list
        print(f"Error using LegalBERT for keyword extraction: {e}")
        return []

def enhance_query_with_llm(user_query: str) -> Dict[str, str]:
    """
    Use a hybrid approach (LegalBERT + LLM) to extract key legal terms and 
    generate multiple targeted search queries.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"main": user_query, "amount_focused": user_query}

    # Step 1: Extract precise keywords with LegalBERT
    legal_keywords = _extract_keywords_with_legalbert(user_query)
    
    # Step 2: Use Groq LLM, enhanced with LegalBERT keywords if available
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)
    
    # Base prompts
    system_prompt = """You are a legal search expert specializing in finding arbitration awards with monetary damages.

Your goal is to generate two distinct search queries in JSON format:
{
  "main_query": "A broad search for the case type and key legal issues.",
  "amount_query": "A specific search focused on monetary outcomes like awards, damages, or compensation."
}

For main_query: Include case type, jurisdiction, and legal issues.
For amount_query: Add terms like "award amount", "damages awarded", "compensation", "INR/USD", "crore/million".

Return ONLY valid JSON, no other text."""

    user_prompt_template = """Case: {user_query}

{keyword_section}Generate optimized search queries based on the provided information (JSON format only)."""

    keyword_section = ""
    if legal_keywords:
        keyword_str = ", ".join(legal_keywords)
        keyword_section = f"Important keywords identified: [{keyword_str}]\n\nUse these keywords as the foundation for your search queries.\n"

    user_prompt = user_prompt_template.format(user_query=user_query, keyword_section=keyword_section)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=200, # Increased slightly for potentially longer queries
        )
        content = response.choices[0].message.content.strip() if response.choices else ""
        
        # Try to parse JSON from the response
        import json
        queries = json.loads(content)
        return {
            "main": queries.get("main_query", user_query),
            "amount_focused": queries.get("amount_query", user_query)
        }
    except Exception:
        # Fallback if LLM or JSON parsing fails
        return {"main": user_query, "amount_focused": user_query}

def call_groq_analysis(query: str, docs: List[Dict[str, Any]]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ GROQ_API_KEY not found. Cannot generate LLM analysis."
    
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)
    
    # Take only top 5 most relevant cases (already sorted by similarity)
    top_docs = docs[:5]
    
    # Prepare detailed context from top 5 documents
    context_parts = []
    for i, d in enumerate(top_docs, 1):
        title = d.get("title", "Untitled")
        url = d.get("url", "")
        text = d.get("text", "")[:8000]
        similarity = d.get("similarity", 0)
        
        context_parts.append(f"""
### CASE {i} (Relevance: {similarity:.1%})
**Title**: {title}
**Source**: {url}
**Full Content**:
{text}
---
""")
    
    context = "\n".join(context_parts)
    
    system_prompt = """You are an expert arbitration analyst with deep knowledge of case law and dispute resolution. Your task is to:
1. **Analyze Each Case Thoroughly**: Read all 5 provided cases carefully. For each case, identify:
   - Parties involved and their relationship
   - Nature of the dispute (construction, commercial, employment, etc.)
   - Key facts and timeline
   - Claims made and amounts sought
   - Legal issues and arguments
   - Final award/outcome (if mentioned)
2. **Draw Detailed Analogies**: Create a comprehensive comparison showing:
   - Which case is MOST similar to the user's situation and why (detailed comparison)
   - Specific parallels in facts, claims, and circumstances
   - Key differences that might affect the outcome
   - Patterns across multiple cases that apply to user's scenario
3. **Estimate Arbitration Amount**: Based on the analyzed cases, provide:
   - A reasonable range or specific estimate
   - Clear explanation of how you arrived at this figure
   - Factors that could increase or decrease the amount
   - Confidence level in your estimate
4. **Strategic Insights**: Provide actionable insights:
   - Precedents that favor/disfavor the claim
   - Critical success factors from similar cases
   - Potential risks and opportunities
Be thorough, specific, and reference exact details from the cases. Use numbers, dates, and facts from the documents."""

    user_prompt = f"""**USER'S ARBITRATION CASE:**
{query}

**TOP 5 MOST RELEVANT CASES (Ranked by AI Similarity):**
{context}
Please provide a comprehensive analysis with detailed analogies to help understand how these cases relate to the user's situation."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
        )
        return response.choices[0].message.content if response.choices else "No response generated."
    except Exception as e:
        return f"⚠️ Error calling Groq API: {str(e)}"
