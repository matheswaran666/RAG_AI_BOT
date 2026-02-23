import secrets
class Utils:
    def __init__(self):
        pass
    
    def generate_api_key(self):
        return secrets.token_hex(32)