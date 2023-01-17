"""
Módulo contendo as funções métricas de performance e risco.
"""

# Importando bibliotecas
import contextlib
import pandas as pd
import numpy as np
from math import ceil as ceil, sqrt
from tabulate import tabulate
from dateutil.relativedelta import relativedelta
from datetime import datetime as dt
import importlib

# Importando funções dos módulos locais
from . import utils
from . import stats
importlib.reload(utils)
importlib.reload(stats)

def compare(returns:pd.Series, benchmark:pd.Series, aggregate:str=None, compounded:bool=True,
            round_vals:int=None)->pd.DataFrame:
    """
    Compare returns to benchmark on a
    day/week/month/quarter/year basis
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series) or not isinstance(benchmark,pd.Series):    
        raise ValueError('Função só aceita data series como input de returns.')
    # Agregando os dasdos por periodo de interesse, se necessário
    data = pd.DataFrame(data={
        'Returns': utils.aggregate_returns(returns, aggregate, compounded) * 100,
        'Benchmark': utils.aggregate_returns(benchmark, aggregate, compounded) * 100})
    # Calculando as estatísticas
    data['Multiplier'] = data['Returns'] / data['Benchmark']
    data['Won'] = np.where(data['Returns'] >= data['Benchmark'], '+', '-')
    return np.round(data, round_vals) if round_vals is not None else data


def monthly_returns(returns:pd.Series, eoy:bool=True, compounded:bool=True)->pd.DataFrame:
    """Calculates monthly returns"""
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input de returns.')

    original_returns = returns.copy()
    returns = pd.DataFrame(utils.group_returns(returns,
                      returns.index.strftime('%Y-%m-01'),
                      compounded))
    returns.columns = ['Returns']
    returns.index = pd.to_datetime(returns.index)
    # get returnsframe
    returns['Year'] = returns.index.strftime('%Y')
    returns['Month'] = returns.index.strftime('%b')
    # make pivot table
    returns = returns.pivot('Year', 'Month', 'Returns').fillna(0)
    # handle missing months
    for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
        if month not in returns.columns:
            returns.loc[:, month] = 0
    # order columns by month
    returns = returns[['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
    if eoy:
        returns['eoy'] = utils.group_returns(
            original_returns, 
            original_returns.index.year, 
            compounded=compounded).values

    returns.columns = map(lambda x: str(x).upper(), returns.columns)
    returns.index.name = None
    return returns

def drawdown_details(returns:pd.Series)->pd.DataFrame:
    """
    Calculates drawdown details, including start/end/valley dates,
    duration, max drawdown and max dd for 99% of the dd period
    for every drawdown period
    """
    # Lidando com exceções
    if not isinstance(returns,pd.Series):    
        raise ValueError('Função só aceita data series como input de returns.')
    # Criando as séries de drawdonw
    drawdown = utils.to_drawdown_series(returns)
    # Extraindo a data de início (primeiro drawdown)
    no_dd = drawdown == 0 # mark no drawdown
    starts = ~no_dd & no_dd.shift(1)
    starts = list(starts[starts].index) # Conjunto de datas a partir do primeiro drawdwon
    # Extraindo a data final (primeiro drawdown)
    ends = no_dd & (~no_dd).shift(1)
    ends = list(ends[ends].index)
    # Caso não tivermos drawdown na série, iremos retornar o dartaframe vazio
    if not starts:
        return pd.DataFrame(
            index=[], columns=('start', 'valley', 'end', 'days',
                                'max drawdown', '99% max drawdown'))
    # drawdown series begins in a drawdown
    if ends and starts[0] > ends[0]:
        starts.insert(0, drawdown.index[0])
    # series ends in a drawdown fill with last date
    if not ends or starts[-1] > ends[-1]:
        ends.append(drawdown.index[-1])
    # Construindo dataframe com as estatísticas
    data = []
    for i, _ in enumerate(starts):
        dd = drawdown[starts[i]:ends[i]]
        clean_dd = -utils.remove_outliers(-dd, .99)
        data.append((starts[i], dd.idxmin(), ends[i],
                    (ends[i] - starts[i]).days,
                     dd.min() * 100, clean_dd.min() * 100))
    df = pd.DataFrame(data=data,columns=('start', 'valley', 'end', 'days',
                                        'max drawdown',
                                        '99% max drawdown'))
    # Preparando o display dos resultados
    df['days'] = df['days'].astype(int)
    df['max drawdown'] = df['max drawdown'].astype(float)
    df['99% max drawdown'] = df['99% max drawdown'].astype(float)
    df['start'] = df['start'].dt.strftime('%Y-%m-%d')
    df['end'] = df['end'].dt.strftime('%Y-%m-%d')
    df['valley'] = df['valley'].dt.strftime('%Y-%m-%d')
    return df

def calc_dd(returns:pd.Series, benchmark:pd.Series=None, display:bool=True, as_pct:bool=False)->pd.DataFrame:
    # Lidando com exceções
    if not isinstance(returns,pd.Series) or not isinstance(benchmark,(pd.Series,type(None))):    
        raise ValueError('Função só aceita data series como input de returns.')
    # Criando a série de drawdowns
    dd_info = drawdown_details(returns)
    if dd_info.empty:
        return pd.DataFrame()
    # Criando as estatísticas de drawdown para as estratégias
    ret_dd = dd_info['returns'] if "returns" in dd_info else dd_info
    dd_stats = {'returns': {
                'Max Drawdown %': ret_dd.sort_values(by='max drawdown', ascending=True)['max drawdown'].values[0] / 100,
                'Longest DD Days': str(np.round(ret_dd.sort_values(by='days', ascending=False)['days'].values[0])),
                'Avg. Drawdown %': ret_dd['max drawdown'].mean() / 100,
                'Avg. Drawdown Days': str(np.round(ret_dd['days'].mean()))}}
    # Criando as estatísticas de drawdown para o benchmark
    if not isinstance(benchmark,type(None)):
        bench_dd = dd_info['benchmark'].sort_values(by='max drawdown')
        dd_stats['benchmark'] = {
            'Max Drawdown %': bench_dd.sort_values(by='max drawdown', ascending=True)['max drawdown'].values[0] / 100,
            'Longest DD Days': str(np.round(bench_dd.sort_values(by='days', ascending=False)['days'].values[0])),
            'Avg. Drawdown %': bench_dd['max drawdown'].mean() / 100,
            'Avg. Drawdown Days': str(np.round(bench_dd['days'].mean()))}
    # Preparando o disply das informações (pct multiplier)
    pct = 100 if display or as_pct else 1
    dd_stats = pd.DataFrame(dd_stats).T
    dd_stats['Max Drawdown %'] = dd_stats['Max Drawdown %'].astype(float) * pct
    dd_stats['Avg. Drawdown %'] = dd_stats['Avg. Drawdown %'].astype(float) * pct
    return dd_stats.T

def metrics(returns:pd.Series, benchmark:pd.Series=None, rf=0., as_pct:bool=True, display:bool=True,
            sep:bool=True, compounded:bool=True, periods_per_year:int=252):  
            # sourcery skip: low-code-quality
    # Lidando com exceções
    if not isinstance(returns,pd.Series) or not isinstance(benchmark,(pd.Series,type(None))):    
        raise ValueError('Função só aceita data series como input de returns.')
    
    # Setando o pct multiplier
    pct = 100 if as_pct else 1
    blank = ['']

    # Primeria parte do relatório será sobre as caracaterísticas das séries analisadas
    s_start = {'returns': returns.index.strftime('%Y-%m-%d')[0]}
    s_end = {'returns': returns.index.strftime('%Y-%m-%d')[-1]}
    if benchmark is not None:
        # Passando para o benchmark, se houver
        s_start['benchmark'] = benchmark.index.strftime('%Y-%m-%d')[0]
        s_end['benchmark'] = benchmark.index.strftime('%Y-%m-%d')[-1]
        blank = ['','']

    # Criando o dataframe de metrics (será utilizdo para o )
    metrics = pd.DataFrame()

    # Adicionando as informações de datas inciais 
    metrics['Start Period'] = pd.Series(s_start)
    metrics['End Period'] = pd.Series(s_end)
    metrics['Time in Market %'] = stats.exposure(returns) * pct

    # Quebra de bloco
    metrics['~~~~~~~~~'] = blank

    # Partindo para o próximo bloco, temos a peformance da estratégia
    metrics['Cumulative Return %'] = (utils.comp(returns) * pct)
    metrics['CAGR﹪%'] = stats.cagr(returns, rf, compounded) * pct
    metrics['Sharpe'] = stats.sharpe(returns, rf, periods_per_year, True)
    metrics['Sortino'] = stats.sortino(returns, rf, periods_per_year, True)
    metrics['Sortino/√2'] = metrics['Sortino'] / sqrt(2)

    # Quebra de bloco
    metrics['~~~~~~~~'] = blank

    # Partindo para o próximo bloco, temos o risco da estratégia
    dd = calc_dd(returns, as_pct=as_pct)
    for ix, row in dd.iterrows():
        metrics[ix] = row
    metrics['Volatility (ann.) %'] = stats.volatility(returns, periods_per_year) * pct
    metrics['Calmar'] = stats.calmar(returns)
    metrics['Skew'] = stats.skew(returns)
    metrics['Kurtosis'] = stats.kurtosis(returns)
    metrics['Daily Value-at-Risk %'] = -abs(stats.var(returns) * pct)
    metrics['Expected Shortfall (cVaR) %'] = -abs(stats.cvar(returns) * pct)

    # Quebra de bloco
    metrics['~~~~~~~'] = blank

    # Organizando o bloco de retotnos do report
    comp_func = utils.comp
    today = returns.index[-1]  # _dt.today()
    metrics['MTD %'] = comp_func(returns[returns.index >= dt(today.year, today.month, 1)]) * pct
    metrics['3M %'] = comp_func(returns[returns.index >= (today - relativedelta(months=3))]) * pct
    metrics['6M %'] = comp_func(returns[returns.index >= (today - relativedelta(months=6))]) * pct
    metrics['YTD %'] = comp_func(returns[returns.index >= dt(today.year, 1, 1)]) * pct
    metrics['1Y %'] = comp_func(returns[returns.index >= (today - relativedelta(years=1))]) * pct
    metrics['All-time (ann.) %'] = stats.cagr(returns, 0., compounded) * pct

    # Quebra de bloco
    metrics['~~~~~~'] = blank

    # Organizando blooco de best/worst do report
    metrics['Best Day %'] = utils.best(returns) * pct
    metrics['Worst Day %'] = utils.worst(returns) * pct
    metrics['Best Month %'] = utils.best(returns, aggregate='M') * pct
    metrics['Worst Month %'] = utils.worst(returns, aggregate='M') * pct
    metrics['Best Year %'] = utils.best(returns, aggregate='A') * pct
    metrics['Worst Year %'] = utils.worst(returns, aggregate='A') * pct

    # Quebra de bloco
    metrics['~~~~~'] = blank

    # Organizando o bloco de win rate do report
    metrics['Avg. Up Month %'] = utils.avg_win(returns, aggregate='M') * pct
    metrics['Avg. Down Month %'] = utils.avg_loss(returns, aggregate='M') * pct
    metrics['Win Days %%'] = utils.win_rate(returns) * pct
    metrics['Win Month %%'] = utils.win_rate(returns, aggregate='M') * pct
    metrics['Win Quarter %%'] = utils.win_rate(returns, aggregate='Q') * pct
    metrics['Win Year %%'] = utils.win_rate(returns, aggregate='A') * pct

    if "benchmark" in returns:
        # Quebra de bloco
        metrics['~~~~'] = blank
        greeks = greeks(returns, benchmark, periods_per_year)
        metrics['Beta'] = [str(round(greeks['beta'], 2)), '-']
        metrics['Alpha'] = [str(round(greeks['alpha'], 2)), '-']
        metrics['Correlation'] = [f'{str(round(benchmark.corr(returns) * pct, 2))}%', '-']


    # Preparando os blocos para o display
    for col in metrics.columns:
        with contextlib.suppress(Exception):
            metrics[col] = metrics[col].astype(float).round(2)
            if display:
                metrics[col] = metrics[col].astype(str)
        if "*int" in col and display:
            metrics[col] = metrics[col].str.replace('.0', '', regex=False)
            metrics.rename({col: col.replace("*int", "")}, axis=1, inplace=True)
        if "%" in col and display:
            metrics[col] = metrics[col].astype(str) + '%'

    # Ajustano os valores das contas de drwdown
    try:
        metrics['Longest DD Days'] = pd.to_numeric(metrics['Longest DD Days']).astype('int')
        metrics['Avg. Drawdown Days'] = pd.to_numeric(metrics['Avg. Drawdown Days']).astype('int')
        if display:
            metrics['Longest DD Days'] = metrics['Longest DD Days'].astype(str)
            metrics['Avg. Drawdown Days'] = metrics['Avg. Drawdown Days'].astype(str)
    except Exception:
        metrics['Longest DD Days'] = '-'
        metrics['Avg. Drawdown Days'] = '-'
        if display:
            metrics['Longest DD Days'] = '-'
            metrics['Avg. Drawdown Days'] = '-'

    # Criando as repertições entre os blocos e ajustando valores em percentual 
    metrics.columns = [col if '~' not in col else '' for col in metrics.columns]
    metrics.columns = [col[:-1] if '%' in col else col for col in metrics.columns]
    metrics = metrics.T

    # Limpando os valores com erros 
    metrics.replace([-0, '-0'], 0, inplace=True)
    metrics.replace([np.nan, -np.nan, np.inf, -np.inf,
                     '-nan%', 'nan%', '-nan', 'nan',
                    '-inf%', 'inf%', '-inf', 'inf'], '-', inplace=True)

    if display:
        print(tabulate(metrics, headers="keys", tablefmt='simple'))
        return None
    if not sep:
        metrics = metrics[metrics.index != '']

    # remove spaces from column names
    metrics = metrics.T
    metrics.columns = [c.replace(' %', '').replace(' *int', '').strip() for c in metrics.columns]
    metrics = metrics.T
    return metrics
