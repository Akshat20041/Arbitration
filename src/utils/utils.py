import os
import torch
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# Environment & Caching
# ---------------------------
load_dotenv()

@st.cache_resource(show_spinner=False)
def get_legal_bert():
    model_name = os.getenv("LEGAL_BERT_MODEL", "nlpaueb/legal-bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_texts(texts: list[str]) -> np.ndarray:
    tokenizer, model = get_legal_bert()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.matmul(a_norm, b_norm.T)
