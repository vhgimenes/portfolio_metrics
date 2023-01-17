"""
Módulo contendo funções auxiliares para os outros módulos do pacote.
"""

# Importando as bibliotecas
import pandas as pd
import datetime as dt 
import numpy as np 
from math import ceil as ceil # Round a number upward to its nearest integer
import pandas as pd
import numpy as np

#! Funções auxilires para trabalhar os retornos
def compsum(returns:pd.Series)->pd.Series:
    """Calculates rolling compounded returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    return returns.add(1).cumprod() - 1

def cumsum(returns:pd.Series)->pd.Series:
    """Calculates cumulative sum of returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    return returns.cumsum()

def comp(returns:pd.Series)->float:
    """Calculates total compounded returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    return returns.add(1).prod() - 1

def sum(returns:pd.Series)->float:
    """Calculates the sum of returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    return np.sum

#! Funções auxiliares para filtragem de retorno das séries
def get_mtd_returns(returns:pd.Series)->pd.Series:
    """Filter mtd returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Filtrando observações do mês corrente
    date = pd.to_datetime(max(returns.index))
    return returns[returns.index >= date.strftime('%Y-%m-01')]

def get_qtd_returns(returns:pd.Series)->pd.Series:
    """Filter qtd returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Filtrando observações do trimestre corrente
    date = pd.to_datetime(max(returns.index))
    q = ceil(date.month/3.)
    months = [1,4,7,10] 
    for count, i in enumerate([1,2,3,4]):
        if i == q: 
            return returns[returns.index >= dt.datetime(
                date.year, months[count], 1).strftime('%Y-%m-01')]

def get_std_returns(returns:pd.Series)->pd.Series:
    """Filter std returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Filtrando observações do semestre corrente
    date = pd.to_datetime(max(returns.index))
    if date.month <= 6:
        return returns[returns.index < dt.datetime(date.year, 7, 1).strftime('%Y-%m-01')]
    else:
        return returns[returns.index >= dt.datetime(date.year, 7, 1).strftime('%Y-%m-01')]

def get_ytd_returns(returns:pd.Series)->pd.Series:
    """Filter ytd returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Filtrando observações do ano corrente
    date = pd.to_datetime(max(returns.index))
    return returns[returns.index >= date.strftime('%Y-01-01')]

def get_specific_returns(returns:pd.Series, dates:list)->pd.Series:
    """Filter specific returns by list of dates"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series) or not isinstance(dates,list):    
        raise ValueError('Função só aceita returns como sendo data series e dates como lista.')
    # Filtrando observações de acordo com input
    return returns[returns.index.isin(dates)]

#! Funções auxilires de preparação das séries de retornos 
def to_excess_returns(returns:pd.Series, rf=0., nperiods:int=None)->pd.Series:
    """
    Calculates excess returns by subtracting
    risk-free returns from total returns
    """
    # Lidando com exceções
    if not isinstance(returns, pd.Series) and (isinstance(rf, (pd.Series, float))):
        raise ValueError('Função só aceita data series ou float para rf como input.')
    # Se quisermos alterar a frequência da taxa de risk-free
    if nperiods is not None:
        # deannualize risk-free
        rf = np.power(1 + rf, 1. / nperiods) - 1.
    # Construindo o retorno em excesso
    return returns - rf

def to_log_returns(returns:pd.Series)->pd.Series:
    """Converts returns series to log returns"""
    # Lidando com exceções
    if not isinstance(returns, pd.Series):
        raise ValueError('Função só aceita data series como input.')
    # Passando os retornos para escala log
    try:
        # Transformando os retornos em base log
        return np.log(returns+1) #.replace([np.inf, -np.inf], float('NaN'))
    except Exception:
        return 0.
    
def prepare_returns(data:pd.Series, rf=0., nperiods:int=None, cumulative:str=None, log:bool=False)->pd.Series:
    # sourcery skip: assign-if-exp, reintroduce-else
    """
    Calculates returns of the prices in data, also excess returns
    and compounded returns:
        1.Calculates rolling compounded returns (compsum)
        2.Calculates total compounded returns (comp)
    """
    # Lidando com exceções
    if (not isinstance(data,(pd.Series))) or (not isinstance(rf, (pd.Series, float))):
        raise ValueError('Função só aceita data series ou float para rf como input.')
    # Aqui calcularemos os retornos das séries
    elif data.min() >= 0 and data.max() > 1:
        # Caso data for uma série de preços, iremos:
        data = data.pct_change()
        print("Calculating the returns of the prices \n")
    # Limpandos as séries de retornos para previnir erros
    data = data.replace([np.inf, -np.inf], float('NaN'))
    # Caso passarmos algum risk-free, calcularemos o retorno em excesso
    if not isinstance(rf, type(None)):
        data =  to_excess_returns(data, rf, nperiods)
        print("Calculating the excess returns \n")
    # Caso passarmos log=True, calcularemos o retorno na base log
    if log:
        data = to_log_returns(data)
        print("Transforming to log returns \n")
    # Caso passarmos algum método em cumulative, calcularemos os retornos compostos
    if cumulative == 'comp':
        # Calculando o retorno composto total
        print("Calculating the total compouded return \n")
        return comp(data)
    elif cumulative == 'compsum':
        # Calculando o retorno composto rolling
        print("Transforming to rolling compouded returns \n")
        return compsum(data)
    else:
        # Retornando os retornos orginais
        return data

def to_drawdown_series(returns:pd.Series, rf=0., nperiods:int=None)->pd.Series:
    """Convert returns series to drawdown series"""
    # Lidando com exceções
    if (not isinstance(returns,(pd.Series))) or (not isinstance(rf, (pd.Series, float))):
        raise ValueError('Função só aceita data series ou float para rf como input.')
    # Calculando a série de retornos compostos
    performance = prepare_returns(returns, rf, nperiods).add(1).cumprod()
    # Calculando o indicador
    dd = performance.div(performance.cummax())-1
    return dd #.replace([np.inf, -np.inf, -0], 0)

def group_returns(returns:pd.Series, groupby, compounded=True)->pd.Series:
    """
    Summarize returns by periods
        group_returns(returns, returns.index.year)
        group_returns(returns, [returns.index.year, returns.index.month])
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Agrupandos os retornos pela janela de interesse (groupby)
    if compounded:
        return returns.groupby(groupby).apply(comp)
    return returns.groupby(groupby).sum()

def aggregate_returns(returns:pd.Series, period:str=None, compounded=True)->pd.Series:
    """
    Aggregates returns based on date periods
    day/week/month/quarter/year/eow/eom/eoq
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Agrupandos os retornos pela janela de interesse (period)
    if period is None or 'day' in period:
        return returns
    index = returns.index
    if 'week' in period:
        return group_returns(returns, index.week, compounded=compounded)
    if 'month' in period:
        return group_returns(returns, index.month, compounded=compounded)
    if 'quarter' in period:
        return group_returns(returns, index.quarter, compounded=compounded)
    if period == "A" or any(x in period for x in ['year', 'eoy', 'yoy']):
        return group_returns(returns, index.year, compounded=compounded)
    if 'eow' in period or period == "W":
        return group_returns(returns, [index.year, index.week],
                             compounded=compounded)
    if 'eom' in period or period == "M":
        return group_returns(returns, [index.year, index.month],
                             compounded=compounded)
    if 'eoq' in period or period == "Q":
        return group_returns(returns, [index.year, index.quarter],
                             compounded=compounded)
    return returns if isinstance(period, str) else group_returns(returns, period, compounded)

#! Funções auxiliares de contagens em séries temporais  
def count_consecutive(data:pd.Series)->int:
    """Counts consecutive data (like cumsum() with reset on zeroes)"""
    return data * (data.groupby((data != data.shift(1)).cumsum()).cumcount() + 1)

#! Funções de estatísticas de retornos
def outliers(returns:pd.Series, quantile:float=.95)->pd.Series:
    """Returns series of outliers by specified quantile"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    return returns[returns > returns.quantile(quantile)].dropna(how='all')

def remove_outliers(returns:pd.Series, quantile:float=.95)->pd.Series:
    """Returns series of returns without the outliers"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    return returns[returns < returns.quantile(quantile)]

#! Funções de análise exploratória sobre os retornos
def max_drawdown(returns:pd.Series)->float:
    """Calculates the maximum drawdown"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input de returns.')
    # Calculando o indicador
    dd = to_drawdown_series(returns)
    return dd.min()

def best(returns:pd.Series, aggregate:str=None, compounded:bool=True)->float:
    """
    Returns the best return of
    day/month/week/quarter/year
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    return aggregate_returns(returns, aggregate, compounded).max()

def worst(returns:pd.Series, aggregate:str=None, compounded:bool=True)->float:
    """
    Returns the worst return of
    day/month/week/quarter/year
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    return aggregate_returns(returns, aggregate, compounded).min()

def consecutive_wins(returns:pd.Series, aggregate:str=None, compounded:bool=True)->int:
    """
    Returns the maximum consecutive wins by
    day/month/week/quarter/year
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    returns = aggregate_returns(returns, aggregate, compounded) > 0
    return count_consecutive(returns).max()

def consecutive_losses(returns:pd.Series, aggregate:str=None, compounded:bool=True)->int:
    """
    Returns the maximum consecutive losses by
    day/month/week/quarter/year
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    returns = aggregate_returns(returns, aggregate, compounded) < 0
    return count_consecutive(returns).max()

def win_rate(returns:pd.Series, aggregate:str=None, compounded:bool=True)->float:
    """
    Calculates the win ratio for a period by
    day/month/week/quarter/year
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Agrupando a série, se aggregate=None retornará a seríe horiginal
    returns = aggregate_returns(returns, aggregate, compounded)
    # Calcula do win_rate
    try:
        return len(returns[returns > 0]) / len(returns[returns != 0])
    except Exception:
        return 0.
    
def avg_return(returns:pd.Series, aggregate:str=None, compounded:bool=True)->float:
    """Calculates the average return/trade return for a period"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Agrupando a série, se necessário 
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return returns[returns != 0].dropna().mean()

def avg_win(returns:pd.Series, aggregate:str=None, compounded:bool=True)->float:
    """
    Calculates the average winning
    return/trade return for a period
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Agrupando a série, se necessário 
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return returns[returns > 0].dropna().mean()

def avg_loss(returns:pd.Series, aggregate:str=None, compounded:bool=True)->float:
    """
    Calculates the average low if
    return/trade return for a period
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Agrupando a série, se necessário 
    if aggregate:
        returns = aggregate_returns(returns, aggregate, compounded)
    return returns[returns < 0].dropna().mean()


    
