import logging
import pathlib
from datetime import datetime
from typing import Optional

class SingletonLogger:
    _instance: Optional['SingletonLogger'] = None
    _logger: Optional[logging.Logger] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
    
    def setup(self, name: str = "GraphDiffusion", log_dir: pathlib.Path = None, level=logging.INFO):
        if self._logger is not None:
            return self._logger
            
        if log_dir is None:
            log_dir = pathlib.Path("logs")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        
        if self._logger.handlers:
            self._logger.handlers.clear()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"{name}_{timestamp}.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self._logger.addHandler(file_handler)
        self._logger.addHandler(console_handler)
        
        return self._logger
    
    def get_logger(self) -> logging.Logger:
        if self._logger is None:
            self.setup()
        return self._logger
    
    @classmethod
    def get_instance(cls) -> 'SingletonLogger':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

def get_logger() -> logging.Logger:
    return SingletonLogger.get_instance().get_logger()

def setup_singleton_logger(name: str = "GraphDiffusion", log_dir: pathlib.Path = None, level=logging.INFO) -> logging.Logger:
    return SingletonLogger.get_instance().setup(name, log_dir, level)
