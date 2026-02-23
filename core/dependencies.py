from fastapi import Request

def get_llm(request: Request):
    if not hasattr(request.app.state, "llm"):
        from core.util_instances import get_llm_instance
        request.app.state.llm = get_llm_instance()
    return request.app.state.llm


def get_utils(request: Request):
    if not hasattr(request.app.state, "utils"):
        from core.util_instances import get_utils_instance
        request.app.state.utils = get_utils_instance()
    return request.app.state.utils