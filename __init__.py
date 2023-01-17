"""
Adicionar as extenções 
https://github.com/ranaroussi/quantstats/blob/main/quantstats/__init__.py
"""
# Importando libs externas
import importlib
# Importando as funções do pacote
from . import utils
from . import stats
from . import graphs
from . import metrics
# Para permitir que mudanças passem a ser usadas diretamente
importlib.reload(utils)
importlib.reload(stats)
importlib.reload(graphs)
importlib.reload(metrics)