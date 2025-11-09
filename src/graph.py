import asyncio
import streamlit as st
from typing import List, Dict, Any

from langgraph.graph import StateGraph, END

from src.state import WorkflowState
from src.analysis.llm_analysis import enhance_query_with_llm, call_groq_analysis
from src.searching.serpapi_search import serpapi_multi_search
from src.crawling.crawler import crawl_all, html_to_text
from src.ranking.ranker import node_rank


def node_search(state: WorkflowState) -> WorkflowState:
    q = state["query"]
    
    # Enhance query using LLM
    st.info("🧠 Analyzing case to generate optimized search queries...")
    queries = enhance_query_with_llm(q)
    state["enhanced_query"] = queries.get("main", q)
    
    st.success(f"✨ **Main Query:** {queries['main'][:100]}...")
    st.success(f"💰 **Amount-Focused Query:** {queries['amount_focused'][:100]}...")
    
    # Multi-source search
    st.info("🔍 Searching across legal databases and case law repositories...")
    results = serpapi_multi_search(queries, n=15)
    state["search_results"] = results
    
    if not results:
        state["error"] = "No search results found."
    else:
        st.success(f"✅ Found {len(results)} results from authenticated sources")
    
    return state

def node_crawl(state: WorkflowState) -> WorkflowState:
    results = state.get("search_results", [])
    docs: List[Dict[str, Any]] = []
    
    if not results:
        state["docs"] = docs
        return state
    
    # Extract URLs and metadata
    url_map = {}
    urls_to_crawl = []
    
    for r in results:
        url = r.get("url")
        if url:
            url_map[url] = {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "score": r.get("score", 0),
                "source": r.get("source", "Unknown")
            }
            urls_to_crawl.append(url)
    
    if not urls_to_crawl:
        state["docs"] = docs
        return state
    
    # Limit to top 15 for crawling
    urls_to_crawl = urls_to_crawl[:15]
    
    st.info(f"🚀 Starting async crawl of {len(urls_to_crawl)} URLs...")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Crawl all URLs concurrently
    try:
        status_text.text("⚡ Fetching pages concurrently...")
        crawled_results = asyncio.run(crawl_all(urls_to_crawl))
        
        # Process results
        for idx, item in enumerate(crawled_results):
            url = item["url"]
            html = item["html"]
            metadata = url_map.get(url, {})
            
            progress_bar.progress((idx + 1) / len(crawled_results))
            status_text.text(f"Processing {idx + 1}/{len(crawled_results)}: {metadata.get('title', '')[:50]}...")
            
            if not html:
                # Store snippet if can't fetch full page
                docs.append({
                    "url": url,
                    "title": metadata.get("title", ""),
                    "text": metadata.get("snippet", ""),
                    "full_fetch": False,
                    "score": metadata.get("score", 0),
                    "source": metadata.get("source", "Unknown")
                })
                continue
            
            # Convert HTML to text
            text = html_to_text(html)
            
            if not text or len(text) < 100:
                docs.append({
                    "url": url,
                    "title": metadata.get("title", ""),
                    "text": metadata.get("snippet", ""),
                    "full_fetch": False,
                    "score": metadata.get("score", 0),
                    "source": metadata.get("source", "Unknown")
                })
                continue
            
            docs.append({
                "url": url,
                "title": metadata.get("title", ""),
                "text": text[:50000],  # Limit text size
                "full_fetch": True,
                "score": metadata.get("score", 0),
                "source": metadata.get("source", "Unknown")
            })
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Successfully crawled {sum(1 for d in docs if d.get('full_fetch'))} pages (full), {sum(1 for d in docs if not d.get('full_fetch'))} snippets")
        
    except Exception as e:
        st.error(f"⚠️ Crawling error: {str(e)}")
        progress_bar.empty()
        status_text.empty()
    
    state["docs"] = docs
    return state


def node_llm_analysis(state: WorkflowState) -> WorkflowState:
    ranked = state.get("ranked", [])
    
    if not ranked:
        state["llm_response"] = "No documents available for analysis."
        return state
    
    query = state["query"]
    llm_response = call_groq_analysis(query, ranked)
    state["llm_response"] = llm_response
    
    return state


def build_graph():
    g = StateGraph(WorkflowState)
    g.add_node("search", node_search)
    g.add_node("crawl", node_crawl)
    g.add_node("rank", node_rank)
    g.add_node("llm_analysis", node_llm_analysis)

    g.set_entry_point("search")
    g.add_edge("search", "crawl")
    g.add_edge("crawl", "rank")
    g.add_edge("rank", "llm_analysis")
    g.add_edge("llm_analysis", END)
    
    return g.compile()
