import os
import streamlit as st

from src.graph import build_graph

# --------------------------- 
# Streamlit App
# --------------------------- 

def main():
    st.set_page_config(page_title="Arbitration Amount Predictor", layout="wide")
    
    st.title("⚖️ Arbitration Amount Predictor")
    st.markdown("""
    Enter your arbitration case details below. The system will:
    1. 🔍 Search for similar cases using SERPAPI
    2. 📄 Crawl and parse relevant documents
    3. 🤖 Analyze using Legal-BERT and Groq LLM
    4. 💰 Provide amount estimation and case insights
    """)
    
    # Check for API keys
    col1, col2 = st.columns(2)
    with col1:
        if os.getenv("SERPAPI_API_KEY"):
            st.success("✅ SERPAPI configured")
        else:
            st.error("❌ SERPAPI_API_KEY missing")
    
    with col2:
        if os.getenv("GROQ_API_KEY"):
            st.success("✅ Groq API configured")
        else:
            st.error("❌ GROQ_API_KEY missing")
    
    st.markdown("---")
    
    query = st.text_area(
        "**Describe your arbitration case:**",
        placeholder="Example: Construction contract dispute with Godrej involving delay claims and cost overruns. Project value around $50M, claimant seeks damages for delays and defective work.",
        height=150,
        help="Provide details about parties, dispute type, claims, project value, etc."
    )
    
    run = st.button("🔍 Analyze Case", type="primary", use_container_width=True)
    
    if run:
        if not query.strip():
            st.warning("⚠️ Please enter a case description.")
            st.stop()
        
        if not os.getenv("SERPAPI_API_KEY"):
            st.error("⚠️ SERPAPI_API_KEY is required. Please set it in your .env file.")
            st.stop()
        
        if not os.getenv("GROQ_API_KEY"):
            st.error("⚠️ GROQ_API_KEY is required. Please set it in your .env file.")
            st.stop()
        
        with st.spinner("🔍 Searching and analyzing cases..."):
            graph = build_graph()
            state = graph.invoke({"query": query})
        
        if state.get("error"):
            st.error(state["error"])
            st.stop()
        
        # Display search results
        st.markdown("---")
        st.subheader("📚 Retrieved Cases")
        
        ranked = state.get("ranked", [])
        if ranked:
            for i, doc in enumerate(ranked, 1):
                score_emoji = "🔥" if doc.get('score', 0) >= 10 else "⭐" if doc.get('score', 0) >= 5 else "📄"
                with st.expander(f"{score_emoji} Case {i}: {doc.get('title', 'Untitled')[:100]}", expanded=(i <= 3)):
                    st.markdown(f"**🔗 URL:** [{doc['url']}]({doc['url']})")
                    st.markdown(f"**🎯 Relevance Score:** {doc.get('similarity', 0):.2%}")
                    st.markdown(f"**📊 Content Score:** {doc.get('score', 0)} points")
                    st.markdown(f"**📁 Source Type:** {doc.get('source', 'Unknown')}")
                    st.markdown(f"**📥 Full Fetch:** {'✅ Yes' if doc.get('full_fetch') else '❌ No (snippet only)'}")
                    
                    # Show preview
                    preview = doc.get('text', '')[:500]
                    st.text_area(f"Preview", preview, height=100, key=f"preview_{i}")
        else:
            st.warning("No cases retrieved.")
        
        # Display LLM analysis
        st.markdown("---")
        st.subheader("🤖 AI Analysis & Estimation")
        
        llm_response = state.get("llm_response", "")
        if llm_response:
            st.markdown(llm_response)
        else:
            st.warning("No analysis generated.")
        
        # Download option
        st.markdown("---")
        if ranked and llm_response:
            report = f"""# Arbitration Case Analysis Report

## Query
{query}

## Retrieved Cases
"""
            for i, doc in enumerate(ranked, 1):
                report += f"\n### Case {i}: {doc.get('title', 'Untitled')}\n"
                report += f"- URL: {doc['url']}\n"
                report += f"- Relevance: {doc.get('similarity', 0):.2%}\n\n"
            
            report += f"\n## AI Analysis\n{llm_response}\n"
            
            st.download_button(
                label="📥 Download Full Report",
                data=report,
                file_name="arbitration_analysis.md",
                mime="text/markdown"
            )
            # Show what query was used
            if state.get("enhanced_query"):
                st.info(f"🔍 **Search Query Used:** {state['enhanced_query']}")

if __name__ == "__main__":
    main()
