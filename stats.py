"""
Módulo contendo funções de indicadores de performance e risco.
"""
# Importando bibliotecas
import pandas as pd
import numpy as np
from math import ceil as ceil
from scipy.stats import norm
import importlib

# Importando módulos locais
from . import utils as utils
importlib.reload(utils)

# Cálculo de Indicadores
def exposure(returns:pd.Series)->float:
    """Returns the market exposure time (returns != 0)"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Cálulo da exposição (tempo do mercado em relação a série temporal) 
    ex = len(returns[(~np.isnan(returns)) & (returns != 0)]) / len(returns)
    return ceil(ex * 100) / 100

def autocorr_penalty(returns:pd.Series)->float:
    """Metric to account for auto correlation"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Cálculo da autocorr
    num = len(returns)
    coef = np.abs(np.corrcoef(returns[:-1], returns[1:])[0, 1])
    corr = [((num - x)/num) * coef ** x for x in range(1, num)]
    return np.sqrt(1 + 2 * np.sum(corr))

def distribution(returns:pd.Series, compounded:bool=True):
    """Returns the distribution of returns by differents windows"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Funções auxiliares de cálculo dos outliers
    def get_outliers(data):
        # https://datascience.stackexchange.com/a/57199
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.
        filtered = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)
        return {"values": data.loc[filtered].tolist(),
                "outliers": data.loc[~filtered].tolist()}
    # Podemos optar por retornos compostos ou mesmo a soma dos retornos
    apply_fnc = utils.comp if compounded else np.sum
    daily = returns.dropna()
    return {"Daily": get_outliers(daily),
            "Weekly": get_outliers(daily.resample('W-MON').apply(apply_fnc)),
            "Monthly": get_outliers(daily.resample('M').apply(apply_fnc)),
            "Quarterly": get_outliers(daily.resample('Q').apply(apply_fnc)),
            "Yearly": get_outliers(daily.resample('A').apply(apply_fnc))}

def volatility(returns:pd.Series, periods_per_year:int=252) -> float:
    # sourcery skip: assign-if-exp, reintroduce-else
    """Calculates the volatility of returns for a period"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Cálculo do desvio padrão da serie 
    std = returns.std()
    # Anualizando para calculo da volatilidade 
    return std * np.sqrt(periods_per_year) # Por default 252

def rolling_volatility(returns:pd.Series, rolling_period:int=126, periods_per_year:int=252)->pd.Series:
    """Calculates the rolling volatility of returns by a period"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Cálculo do desvio padrão rolling da serie 
    res = returns.rolling(rolling_period).std() 
    # Anualizando para calculo da volatilidade
    res = res * np.sqrt(periods_per_year) # Por default 252
    return res.dropna()

# ======= METRICS =======

def sharpe(returns:pd.Series, rf=0., nperiods:int=None, periods_per_year:int=252, smart:bool=False)->float:
    # sourcery skip: assign-if-exp, reintroduce-else
    """
    Calculates the sharpe ratio of access returns
    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    Args:
        * returns (Series, DataFrame): Input return series
        * rf (float): Risk-free rate expressed as a yearly (annualized) return
        * nperiods (int): Freq. of returns (252/365 for daily, 12 for monthly)
        * annualize: return annualize sharpe?
        * smart: return smart sharpe ratio
    """
    # Lidando com exceções
    if not isinstance(returns, pd.Series) and (isinstance(rf, (pd.Series, float))):
        raise ValueError('ValueError: Função só aceita data series ou float para rf como input.')
    # Descontando o risk-free dos retornos
    returns = utils.prepare_returns(returns, rf, nperiods)
    # Tirando o desvio padrão 
    divisor = returns.std(ddof=1)
    # Corrgindo pela autocorr, se necessário
    if smart:
        # penalize sharpe with auto correlation
        divisor = divisor * autocorr_penalty(returns)
    # Criando o indicador
    res = returns.mean() / divisor
    # Anualziando o indicador
    return res * np.sqrt(periods_per_year) # Por default utilizamos 252
    
def rolling_sharpe(returns:pd.Series, rf=0., nperiods:int=None, rolling_period:int=126, periods_per_year:int=252)->pd.Series:
    # Lidando com exceções
    if not isinstance(returns, pd.Series) and (isinstance(rf, (pd.Series, float))):
        raise ValueError('Função só aceita data series ou float para rf como input.')
    # Descontando o risk-free dos retornos
    returns = utils.prepare_returns(returns, rf, nperiods)
    # Criando o indicador
    res = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()
    # Anualziando o indicador
    res = res * np.sqrt(periods_per_year) # Por default utilizamos 252
    return res.dropna() 

def sortino(returns:pd.Series, rf=0., nperiods:int=None, periods_per_year:int=252, smart=False)->float:
    # sourcery skip: assign-if-exp, reintroduce-else
    """
    Calculates the sortino ratio of access returns
    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    Calculation is based on this paper by Red Rock Capital
    http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    # Lidando com exceções
    if not isinstance(returns, pd.Series) and (isinstance(rf, (pd.Series, float))):
        raise ValueError('Função só aceita data series ou float para rf como input.')
    # Descontando o risk-free dos retornos
    returns = utils.prepare_returns(returns, rf, nperiods)
    # Calculando o downside da série
    downside = np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))
    # Ajustando pela autocorr
    if smart:
        # penalize sortino with auto correlation
        downside = downside * autocorr_penalty(returns)
    # Criando o indicador
    res = returns.mean() / downside
    # Anualizando o indicador
    res * np.sqrt(periods_per_year) # Por default utilizamos 252
    return res

def rolling_sortino(returns:pd.Series, rf=0., nperiods:int=None, rolling_period:int=126, periods_per_year:int=252)->pd.Series:
    # Lidando com exceções
    if not isinstance(returns, pd.Series) and (isinstance(rf, (pd.Series, float))):
        raise ValueError('Função só aceita data series ou float para rf como input.')
    # Descontando o risk-free dos retornos
    returns = utils.prepare_returns(returns, rf, nperiods)
    # Calculando o downside da série
    downside = returns.rolling(rolling_period).apply(
        lambda x: (x.values[x.values < 0]**2).sum()) / rolling_period
    # Criando o indicador
    res = returns.rolling(rolling_period).mean() / np.sqrt(downside)
    # Anualizando o indicador
    res = res * np.sqrt(periods_per_year) # Por default 252
    return res.dropna()

def cagr(returns:pd.Series, rf=0., nperiods:int=None, compounded:bool=True)->float:
    """
    Calculates the communicative annualized growth return
    (CAGR%) of access returns
    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    # Lidando com exceções
    if not isinstance(returns, pd.Series) and (isinstance(rf, (pd.Series, float))):
        raise ValueError('Função só aceita data series ou float para rf como input.')
    # Descontando o risk-free dos retornos e calculando o retorno composto
    total = utils.prepare_returns(returns, rf, nperiods)
    total = utils.comp(total) if compounded else np.sum(total)
    # Tempo da série em anos
    years = (returns.index[-1] - returns.index[0]).days / 365.
    # Construindo o indicador
    res = abs(total + 1.0) ** (1.0 / years) - 1
    return res

def skew(returns:pd.Series)->float:
    """
    Calculates returns' skewness
    (the degree of asymmetry of a distribution around its mean)
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Calculando o indicador
    return returns.skew()

def kurtosis(returns:pd.Series)->float:
    """
    Calculates returns' kurtosis
    (the degree to which a distribution peak compared to a normal distribution)
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Calculando o indicador
    return returns.kurtosis()

def value_at_risk(returns:pd.Series, sigma:float=1., confidence:float=0.95)->float:
    # sourcery skip: aug-assign
    """
    Calculats the daily value-at-risk
    (variance-covariance calculation with confidence n)
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Calculando o retorno médio
    mu = returns.mean()
    # Calculando o desvio padrão
    sigma *= returns.std()
    # Calculando o indicador 
    if confidence > 1:
        confidence = confidence/100
    return norm.ppf(1-confidence, mu, sigma)

def var(returns:pd.Series, sigma:float=1., confidence:float=0.95)->float:
    """Shorthand for value_at_risk()"""
    return value_at_risk(returns, sigma, confidence)

def conditional_value_at_risk(returns:pd.Series, sigma:float=1., confidence:float=0.95)->float:
    """
    Calculats the conditional daily value-at-risk (aka expected shortfall)
    quantifies the amount of tail risk an investment
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Calculando o VaR
    var = value_at_risk(returns, sigma, confidence)
    # Calculando o indicado a partir do VaR
    c_var = returns[returns < var].values.mean()
    return c_var if ~np.isnan(c_var) else var

def cvar(returns:pd.Series, sigma:float=1., confidence:float=0.95)->float:
    """Shorthand for conditional_value_at_risk()"""
    return conditional_value_at_risk(returns, sigma, confidence)

def risk_return_ratio(returns:pd.Series)->float:
    """
    Calculates the return / risk ratio
    (sharpe ratio without factoring in the risk-free rate)
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input.')
    # Calculando o indicador
    return returns.mean() / returns.std()

def rolling_risk_return_ratio(returns:pd.Series, rolling_period:int=126, periods_per_year:int=252)->pd.Series:
    """Calculates rolling return / risk ratio (sharpe ratio without factoring in the risk-free rate)"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input de returns.')
    # Criando o indicador
    res = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()
    # Anualizando o indicador
    res = res * np.sqrt(periods_per_year) # Por default 252
    return res.dropna()

def calmar(returns:pd.Series)->float:
    """Calculates the calmar ratio (CAGR% / MaxDD%)"""
    # Calcualdno CAGR
    cagr_ratio = cagr(returns)
    # Calculando o DD Máximo
    max_dd = utils.max_drawdown(returns)
    # Calculando o indicdor
    return cagr_ratio / abs(max_dd)

# ==== VS. BENCHMARK ====

def greeks(returns:pd.Series, benchmark:pd.Series, periods=252.)->float:
    """Calculates alpha and beta of the portfolio"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input de returns.')
    # Calculando a covariância
    matrix = np.cov(returns, benchmark)
    beta = matrix[0, 1] / matrix[1, 1]
    # Calculando o indicador
    alpha = returns.mean() - beta * benchmark.mean()
    alpha = alpha * periods
    return pd.Series({
        "beta":  beta,
        "alpha": alpha,
    }).fillna(0)

