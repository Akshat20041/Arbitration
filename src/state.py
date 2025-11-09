from typing import TypedDict, List, Dict, Any, Optional

class WorkflowState(TypedDict, total=False):
    query: str
    enhanced_query: str
    search_results: List[Dict[str, str]]
    docs: List[Dict[str, Any]]
    ranked: List[Dict[str, Any]]
    llm_response: str
    error: Optional[str]
