import signal
from functools import wraps

class TimeoutError(Exception):
    pass

def timeout(seconds=10):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError("Function call timed out")
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator