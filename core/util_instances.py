from core.llm_utils import llm
from core.utils import Utils

def get_llm_instance():
    return llm()

def get_utils_instance():
    return Utils()

def get_rag_instance():
    return rag()