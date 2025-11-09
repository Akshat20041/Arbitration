from src.state import WorkflowState
from src.utils.utils import embed_texts, cosine_sim

def node_rank(state: WorkflowState) -> WorkflowState:
    q = state["query"]
    docs = state.get("docs", [])
    
    if not docs:
        state["ranked"] = []
        return state
    
    # Embed query and docs
    q_emb = embed_texts([q])
    doc_texts = [d["text"][:2048] for d in docs]
    doc_embs = embed_texts(doc_texts)
    sims = cosine_sim(q_emb, doc_embs).flatten()
    
    for d, s in zip(docs, sims):
        d["similarity"] = float(s)
    
    docs_sorted = sorted(docs, key=lambda x: x.get("similarity", 0.0), reverse=True)
    state["ranked"] = docs_sorted
    
    return state
