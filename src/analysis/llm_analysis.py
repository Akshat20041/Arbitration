import os
import json
from typing import Dict, List, Any
from groq import Groq

def enhance_query_with_llm(user_query: str) -> Dict[str, str]:
    """Use LLM to extract key legal terms and generate multiple targeted search queries"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"main": user_query, "amount_focused": user_query}
    
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)
    
    system_prompt = """You are a legal search expert specializing in finding arbitration awards with monetary damages.

Extract and return TWO search queries in JSON format:
{
  "main_query": "broad search for case type and legal issues",
  "amount_query": "specific search focusing on awards, damages, compensation amounts"
}

For main_query: Include case type, jurisdiction, legal issues
For amount_query: Add terms like "award amount", "damages awarded", "compensation", "INR/USD", "crore/million"

Return ONLY valid JSON, no other text."""

    user_prompt = f"""Case: {user_query}

Generate optimized search queries (JSON format only)."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        content = response.choices[0].message.content.strip() if response.choices else ""
        
        # Try to parse JSON
        import json
        queries = json.loads(content)
        return {
            "main": queries.get("main_query", user_query),
            "amount_focused": queries.get("amount_query", user_query)
        }
    except Exception:
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
        text = d.get("text", "")[:8000]  # ~8k chars per doc = ~2k tokens, 5 docs = ~10k tokens
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
            max_tokens=4000,  # Allow detailed response
        )
        return response.choices[0].message.content if response.choices else "No response generated."
    except Exception as e:
        return f"⚠️ Error calling Groq API: {str(e)}"
