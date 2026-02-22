from core.llm_utils import llm
from core.utils import Utils


llm_instance = llm()
utils_instance = Utils()
rag_instance = llm_instance.rag
