
class FunctionCalling:
    def __init__(self):
        pass
    def load_model(provider, model_name):

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name)

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name)

        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name)

        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            return ChatOllama(model=model_name)

        else:
            raise ValueError("Unsupported model provider")
