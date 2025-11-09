import os
import requests
import streamlit as st
from typing import List, Dict

def serpapi_multi_search(queries: Dict[str, str], n: int = 15) -> List[Dict[str, str]]:
    """Perform multiple searches targeting different sources"""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        st.error("⚠️ SERPAPI_API_KEY not found!")
        return []
    
    all_results = []
    seen_urls = set()
    
    # Priority legal/arbitration databases and sources
    priority_sites = [
        "site:jusmundi.com OR site:italaw.com OR site:iccwbo.org",
        "site:kluwer.com OR site:sccinstitute.com OR site:hcourt.gov.in",
        "site:siac.org.sg OR site:diac.in OR site:niac.in",
        "site:arbitrationindia.com OR site:manupatra.com OR site:sci.gov.in"
    ]
    
    search_configs = [
        # Search 1: Amount-focused with priority sites
        {
            "query": f"{queries['amount_focused']} award damages compensation (site:jusmundi.com OR site:italaw.com OR site:arbitrationindia.com)",
            "label": "Award-focused (Legal DBs)"
        },
        # Search 2: Main query with case law sites
        {
            "query": f"{queries['main']} arbitration award final decision (site:manupatra.com OR site:sci.gov.in OR site:hcourt.gov.in)",
            "label": "Case Law Sites"
        },
        # Search 3: Amount-specific terms
        {
            "query": f"{queries['main']} \"awarded\" \"crore\" OR \"million\" OR \"USD\" OR \"INR\" arbitration",
            "label": "Amount-specific"
        },
        # Search 4: News and analysis sites
        {
            "query": f"{queries['main']} arbitration settlement award (site:barandbench.com OR site:livelaw.in OR site:scobserver.in)",
            "label": "Legal News"
        }
    ]
    
    for config in search_configs[:3]:  # Use top 3 searches to stay within limits
        try:
            params = {
                "engine": "google",
                "q": config["query"],
                "api_key": api_key,
                "num": 10,
            }
            
            st.caption(f"🔍 {config['label']}: {config['query'][:80]}...")
            
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                for item in data.get("organic_results", []) or []:
                    link = item.get("link")
                    title = item.get("title") or ""
                    snippet = item.get("snippet") or ""
                    
                    if link and link not in seen_urls:
                        # Score based on keywords in title/snippet
                        score = 0
                        content_lower = (title + " " + snippet).lower()
                        
                        # High value indicators
                        if any(x in content_lower for x in ["award", "awarded", "damages", "compensation"]):
                            score += 3
                        if any(x in content_lower for x in ["crore", "million", "billion", "usd", "inr", "₹", "$"]):
                            score += 5
                        if any(x in content_lower for x in ["tribunal", "arbitration", "arbitral"]):
                            score += 2
                        if any(x in content_lower for x in ["final", "decision", "order", "judgment"]):
                            score += 2
                        
                        # Authentic source bonus
                        if any(domain in link for domain in ["jusmundi", "italaw", "manupatra", "sci.gov", "hcourt", "arbitration"]):
                            score += 4
                        
                        all_results.append({
                            "title": title,
                            "url": link,
                            "snippet": snippet,
                            "score": score,
                            "source": config["label"]
                        })
                        seen_urls.add(link)
                        
        except Exception as e:
            st.warning(f"Search {config['label']} failed: {str(e)}")
            continue
    
    # Sort by score and return top results
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results[:n]


def serpapi_search(query: str, enhanced_query: str, n: int = 10) -> List[Dict[str, str]]:
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        st.error("⚠️ SERPAPI_API_KEY not found in environment variables!")
        return []
    
    results: List[Dict[str, str]] = []
    try:
        # Use the LLM-enhanced query with additional arbitration keywords
        search_query = f"{enhanced_query} arbitration award case dispute damages"
        
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": api_key,
            "num": min(n, 10),
        }
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            for item in data.get("organic_results", []) or []:
                link = item.get("link")
                title = item.get("title") or ""
                snippet = item.get("snippet") or ""
                if link:
                    results.append({"title": title, "url": link, "snippet": snippet})
                if len(results) >= n:
                    break
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []
    
    return results[:n]
