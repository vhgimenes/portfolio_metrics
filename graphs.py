"""
Módulo contendo funções gráficas do pacote.
"""

import contextlib
# Importanso bibliotecas
from warnings import warn
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import seaborn as sns

# Importando funções dos módulos locais
from . import utils
from . import stats
from . import metrics
importlib.reload(utils)
importlib.reload(stats)
importlib.reload(metrics)

# Instanciando as cores
# ---------------- Retornos e Performances Históricas
def numfmt(x, pos):
    s = f'{x*100:,.1f}' 
    return s

def historical_performance_graph(strategy_dict:dict,  benchmark_list:dict=None, rf_performance:pd.Series=None,
                                 length:int=10, height:int=5):
    # sourcery skip: merge-list-append, move-assign-in-block
    '''
    Plota o gráfico Retorno vs Benchmark

    params: 
        strategy_performance: dict contendo a série da estratégia e seu nome
        index_performance: dict contendo até 9 séries de benchmark e seus respectivos nomes 
        length: comprimento do gráfico 
        height: altura do gráfico      
    '''
    # Extraindo as informações da série principal
    strategy_name = strategy_dict.get(0).get('name')
    strategy_color = strategy_dict.get(0).get('color')
    strategy_perf = strategy_dict.get(0).get('series')

    # Lidando com erros de input
    if not isinstance(strategy_perf, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')

    # Extraindo as datas finais e iniciais da análise para plotagem no gráfico
    init_date = strategy_perf.index.min().strftime('%d-%m-%Y')
    final_date = strategy_perf.index.max().strftime('%d-%m-%Y')

    # Criando a figura que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)

    # Criando lista para guradar os indicadores que irão compor a legenda do gráfico
    perf_list = []
    # PLotando a série principal.
    ax.plot(strategy_perf, linewidth=2, label=strategy_name, color=strategy_color)
    perf_list.append(f'{strategy_name} = {round(strategy_perf[-1]*100,2)}')

    # Checando se a lista com os dicts de benchmarks foi informado no input
    if not isinstance(benchmark_list, type(None)) and isinstance(benchmark_list, dict):
        # Caso passarmos os dicts de benchmarks
        for i in benchmark_list:
            # Extraindo as informações das séries de benchmarks
            bench_name = benchmark_list.get(i).get('name')
            bench_color = benchmark_list.get(i).get('color')
            bench_perf = benchmark_list.get(i).get('series') 
            # Só plataremos benchmarks que forem series
            if isinstance(bench_perf, pd.Series):
                # PLotando cada um dos benchmarks
                ax.plot(bench_perf, linewidth=2, label=bench_name, color=bench_color)
                perf_list.append(f'{bench_name} = {round(bench_perf[-1]*100,2)}') 
            else:
                warn(f'ValueError: series do benchmark {bench_name} não é um data series.')

    # Checando se o risk-free foi informado               
    if not isinstance(rf_performance,type(None)) and isinstance(rf_performance, pd.Series):
            # Plotando o risk-free, se necessário
            ax.plot(rf_performance, linewidth=1, linestyle='--', label='Risk-free', color='red') 
            perf_list.append(f'Rf = {round(rf_performance[-1]*100,2)}') 
            # Colorindo a integral do riks-free
            ax.fill_between(rf_performance.index,rf_performance,color='purple',alpha=0.1, interpolate=True)
            # ax.fill_between(strategy_performance.index,strategy_performance,rf_performance,where=strategy_performance>=rf_performance,color='green',alpha=0.1, interpolate=True)
            # ax.fill_between(strategy_performance.index,strategy_performance,rf_performance,where=strategy_performance<=rf_performance,color='red',alpha=0.1, interpolate=True)

    # Plotando a linha do eixo x (Y=0)
    ax.axhline(y=0, color='black')

    # Criando e formatando os labels do gráfico
    ax.set_xlabel(f'Metrics (%): {" | ".join(perf_list)}', fontsize=13, color = 'darkblue', loc='right', labelpad=10) # Add an x-label to the axes.
    ax.set_ylabel('Historical Performance (%)', fontsize=12, color = 'black') # Add a y-label to the axes.
    ax.tick_params(colors='black', which='both') # Colorindo os ticks dos eixos

    # Colorindo os eixos do gráfico
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # Colorindo o backgroun do gráfico
    ax.set_facecolor('lightgray')
    # Inseirndo grids
    ax.grid(True)

    # Inserindo a legenda no gráfico e colorindo o texto para branco
    ax.legend(loc='best', facecolor='black', fancybox=True, framealpha=0.5)
    [text.set_color("white") for text in ax.legend.get_texts()] # colorindo cada um dos textos 
    # Inserindo título e sub-título e formatando
    plt.suptitle('Historical Performance Analysis',horizontalalignment='center', fontstyle= 'italic', fontsize=14, color='black')
    plt.title(f'Period: {init_date} - {final_date}',horizontalalignment='center', fontsize=12, color='darkblue')

    # Ajustando os eixos y para porcentagens
    ax.yaxis.set_major_formatter(numfmt)
    plt.show()

def historical_daily_returns_graph(strategy_dict:dict, benchmark_list:dict=None, strategy_devs:bool=False, 
                                   length:int=10, height:int=5, labelpad_size:int=10):
    # sourcery skip: extract-method
    """Função responsável pela geração do gráfico de retornos diários da estratégia e dos benchmarks."""
    # Extraindo as informações da série principal
    strategy_name = strategy_dict.get(0).get('name')
    strategy_color = strategy_dict.get(0).get('color')
    strategy_ret = strategy_dict.get(0).get('series')

    # Lidando com erros de input
    if not isinstance(strategy_ret, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')

    # Extraindo as datas finais e iniciais da análise para plotagem no gráfico
    init_date = strategy_ret.index.min().strftime('%d-%m-%Y')
    final_date = strategy_ret.index.max().strftime('%d-%m-%Y')

    # Extraindo o desvio padrão da série de retornos
    std = strategy_ret.std()
    mean = strategy_ret.mean()

    # Criando a figura que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)

    # Plot some data on the axes.
    ax.plot(strategy_ret, linewidth=2, label=strategy_name, color=strategy_color)
    ax.axhline(y=mean, color=strategy_color, linestyle='--')

    # Plotando os desvios do retorno da estratégia no gráfico
    if strategy_devs:
        # PLotando as linhas para visualização das estatísticas
        ax.axhline(y=std, label='+1 std', color='navy', linestyle='--')
        ax.axhline(y=std*2, label='+2 std', color='darkgreen', linestyle='--')
        ax.axhline(y=-std, color='navy', linestyle='--') # label= f'{strategy_name} -1 std', 
        ax.axhline(y=-std*2, color='darkgreen',linestyle='--') # label= f'{strategy_name} -2 std',
        # Ajustando a legenda com as informações extras
        dr_list = [f'{strategy_name}:{round(strategy_ret[-1] * 100, 2)}% (Mean:{round(mean * 100, 2)}%, +1 std:{round(std * 100, 2)}%, +2 std:{round(std*2*100,2)}%)']
    else:
        dr_list = [f'{strategy_name}:{round(strategy_ret[-1] * 100, 2)}% ({round(mean * 100, 2)}%)']

    # Checando se a lista com os dicts de benchmarks foi informado no input
    if not isinstance(benchmark_list, type(None)) and isinstance(benchmark_list, dict):
        # Caso passarmos os dicts de benchmarks
        for i in benchmark_list:
            # Extraindo as informações das séries de benchmarks
            bench_name = benchmark_list.get(i).get('name')
            bench_color = benchmark_list.get(i).get('color')
            bench_ret = benchmark_list.get(i).get('series')
            # Só plataremos benchmarks que forem series
            if isinstance(bench_ret, pd.Series):
                # PLotando cada um dos benchmarks
                bench_ret_mean = bench_ret.mean() 
                ax.plot(bench_ret, linewidth=2, label=bench_name, color=bench_color)
                ax.axhline(y=bench_ret_mean, color=bench_color, linestyle='--')
                dr_list.append(f'{bench_name}: {round(bench_ret[-1]*100,2)}%')

    # Plotando a linha do eixo x (Y=0)
    ax.axhline(y=0, color='black')

    # Criando e formatando os labels do gráfico
    ax.set_xlabel(f'{" | ".join(dr_list)}', fontsize=13, color = 'darkblue', loc='right', labelpad=labelpad_size) # Add an x-label to the axes.
    ax.set_ylabel('Historical Daily Returns (%)', fontsize=12, color = 'black') # Add a y-label to the axes.
    ax.tick_params(colors='black', which='both') # Colorindo os ticks dos eixos

    # Colorindo os eixos do gráfico
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')

    # Colorindo o backgroun do gráfico
    ax.set_facecolor('lightgray')
    # Inseirndo grids
    ax.grid(True)

    # Inserindo a legenda no gráfico e colorindo o texto para branco
    ax.legend(loc='best', facecolor='black', fancybox=True, framealpha=0.5)
    [text.set_color("white") for text in ax.legend.get_texts()] # colorindo cada um dos textos 
    # Inserindo título e sub-título e formatando
    plt.suptitle('Historical Daily Returns Analysis',horizontalalignment='center', fontstyle= 'italic', fontsize=14, color='black')
    plt.title(f'Period: {init_date} - {final_date}',horizontalalignment='center', fontsize=12, color='darkblue')

    # Ajustando o eixos y para porcentagens
    ax.yaxis.set_major_formatter(numfmt)
    plt.show()


# --------------------- Underwater Drawdown

def historical_underwater_graph(strategy_dict:dict, benchmark_dict:dict=None, 
                                length=10, height=5, labelpad_size:int=10):
    # sourcery skip: merge-list-append, merge-list-appends-into-extend, merge-list-extend, move-assign-in-block, simplify-boolean-comparison, unwrap-iterable-construction
    '''
    Essa função plota o gráfico do drawdown

    params: 
        strategy_performance: dataframe com a cota da estratégia 
        index_performance: dataframe com a "cota" do indice
        length: comprimento do gráfico 
        height: altura do gráfico  
    '''
    # Extraindo as informações da série principal
    strategy_name = strategy_dict.get(0).get('name')
    strategy_color = strategy_dict.get(0).get('color')
    strategy_dd = strategy_dict.get(0).get('series')

    # Lidando com erros de input
    if not isinstance(strategy_dd, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')
    
    # Extraindo as datas finais e iniciais da análise para plotagem no gráfico
    init_date = strategy_dd.index.min().strftime('%d-%m-%Y')
    final_date = strategy_dd.index.max().strftime('%d-%m-%Y')

    # Calculando o DD médio
    underwater_mean = strategy_dd.mean()
    
    # Criando lista para guradar os indicadores
    dd_list = [] 
    # Armazenando as métricas
    dd_list.append(f'{strategy_name}:{round(strategy_dd[-1]*100,2)}% ({round(underwater_mean*100,2)}%)')
    
    # Criando a figura que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)
    # Plotando a série principal
    ax.plot(strategy_dd, label=strategy_name, color=strategy_color)
    # Plotando a média histórica do dd da estratégia 
    ax.axhline(y=underwater_mean, color=strategy_color,linestyle='--')
    # Colorindo a integral da série de dd para a estratégia
    ax.fill_between(strategy_dd.index, strategy_dd, color='red',alpha=0.2)
    
    # Checando se a lista com os dicts de benchmarks foi informado no input
    if not isinstance(benchmark_dict, type(None)) and isinstance(benchmark_dict, dict):
        # Caso passarmos os dicts de benchmarks
        for i in benchmark_dict:
            # Extraindo as informações das séries de benchmarks
            bench_name = benchmark_dict.get(i).get('name')
            bench_color = benchmark_dict.get(i).get('color')
            bench_dd = benchmark_dict.get(i).get('series')
            # Só plataremos benchmarks que forem series
            if isinstance(bench_dd, pd.Series):
                # Calculando o dd médio
                bench_dd_mean = bench_dd.mean()
                # Plotando a série de drawdown
                ax.plot(bench_dd, label=bench_name, color=bench_color)
                # Plotando linha com o dd médio
                ax.axhline(y=bench_dd_mean, color=bench_color,linestyle='--')
                # Armazenando as métricas
                dd_list.append(f'{bench_name}:{round(bench_dd[-1]*100,2)}% ({round(bench_dd_mean*100,2)}%)')
    
    # Plotando a linha do eixo x (Y=0)
    ax.axhline(y=0, color='black')
    
    # Criando e formatando os labels do gráfico
    ax.set_xlabel(f'{" | ".join(dd_list)}', fontsize=13, color = 'darkblue', loc='right', labelpad=labelpad_size) # Add an x-label to the axes.
    ax.set_ylabel('Historical Drawdown (%)', fontsize=12, color = 'black') # Add a y-label to the axes.
    ax.tick_params(colors='black', which='both') # Colorindo os ticks dos eixos
    
    # Colorindo os eixos do gráfico
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # Colorindo o backgroun do gráfico
    ax.set_facecolor('lightgray')
    # Inseirndo grids
    ax.grid(True)
    
    # Ajustando o eixos y para porcentagens
    ax.yaxis.set_major_formatter(numfmt)
    
    # Inserindo a legenda no gráfico e colorindo o texto para branco
    ax.legend(loc='best', facecolor='black', fancybox=True, framealpha=0.5)
    [text.set_color("white") for text in ax.legend.get_texts()] # colorindo cada um dos textos 
    # Inserindo título e sub-título e formatando
    plt.suptitle('Historical Drawdown Analysis',horizontalalignment='center', fontstyle= 'italic', fontsize=14, color='black')
    plt.title(f'Period: {init_date} - {final_date}',horizontalalignment='center', fontsize=12, color='darkblue')
    
    # Ajustando os eixos x para porcentagens
    ax.yaxis.set_major_formatter(numfmt)
    plt.show()

# ------------------ Distribuiçao normal dos retornos diários

def histogram_returns_graph(strategy_dict:dict, benchmark_list:dict=None, strategy_stats:bool=False, 
                            length:int=10, height:int=5, bins:int=10, labelpad_size:int=10):
    '''
    Essa função plota o gráfico da distribuição normal dos retornos

    params: 
        strategy_returns: dataframe com retornos diários da estratégia
        length: comprimento do gráfico 
        height: altura do gráfico  
    '''
    # Extraindo as informações da série principal
    strategy_name = strategy_dict.get(0).get('name')
    strategy_color = strategy_dict.get(0).get('color')
    strategy_ret = strategy_dict.get(0).get('series')
    
    # Lidando com erros de input
    if not isinstance(strategy_ret, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')
    
    # Extraindo as datas finais e iniciais da análise para plotagem no gráfico
    init_date = strategy_ret.index.min().strftime('%d-%m-%Y')
    final_date = strategy_ret.index.max().strftime('%d-%m-%Y')
    
    ret_list = []
    # Construindo as métricas 
    mean_return = strategy_ret.mean()
    max_return = strategy_ret.max()
    min_return = strategy_ret.min()
    
    # Criando a figura que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)
    
    # Plotando o histograma
    ax.hist(strategy_ret, bins=bins, label=strategy_name, edgecolor='grey', color=strategy_color)
    plt.axvline(x=mean_return, linestyle='--', linewidth=2, color=strategy_color)
    
    # Plotando o eixo x (Y=0)
    plt.axvline(x=0, linestyle='--', linewidth=2, color='black')
    if strategy_stats:
        # PLotando os indicadores dentro do gráfico
        plt.axvline(x=max_return, linestyle='--', linewidth=2, label='Max', color='darkgreen')
        plt.axvline(x=min_return, linestyle='--', linewidth=2, label='Min', color='darkred')
        ret_list.append(f'{strategy_name}: {round(mean_return*100, 2)} (Max:{round(max_return*100,2)}%, Min:{round(min_return*100,2)}%)')
    else:
        ret_list.append(f'{strategy_name}: {round(mean_return*100, 2)}%')
    
    # Checando se a lista com os dicts de benchmarks foi informado no input
    if not isinstance(benchmark_list, type(None)) and isinstance(benchmark_list, dict):
        # Caso passarmos os dicts de benchmarks
        for i in benchmark_list:
            # Extraindo as informações das séries de benchmarks
            bench_name = benchmark_list.get(i).get('name')
            bench_color = benchmark_list.get(i).get('color')
            bench_ret = benchmark_list.get(i).get('series')
            # Só plataremos benchmarks que forem series
            if isinstance(bench_ret, pd.Series):
                # Calculando o dd médio
                bench_ret_mean = bench_ret.mean()
                # Plotando a série de drawdown
                ax.hist(bench_ret, bins=bins, label=bench_name, edgecolor='grey', color=bench_color, alpha=0.3)
                # Plotando linha com o ret médio
                plt.axvline(x=bench_ret_mean, linestyle='--', linewidth=2, color=bench_color, alpha=1)
                # Armazenando as métricas
                ret_list.append(f'{bench_name}: {round(bench_ret_mean*100, 2)}%')
    
    # Criando e formatando os labels do gráfico
    ax.set_xlabel(f'{" | ".join(ret_list)}', fontsize=13, color = 'darkblue', loc='right', labelpad=labelpad_size) # Add an x-label to the axes.
    ax.set_ylabel('Frequency', fontsize=12, color = 'black') # Add a y-label to the axes.
    ax.tick_params(colors='black', which='both') # Colorindo os ticks dos eixos
    
    # Colorindo os eixos do gráfico
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # Colorindo o backgroun do gráfico
    ax.set_facecolor('lightgray')
    # Inseirndo grids
    ax.grid(True)
    # Ajustando o eixos x para porcentagens
    ax.xaxis.set_major_formatter(numfmt)
    
    # Inserindo a legenda no gráfico e colorindo o texto para branco
    ax.legend(loc='best', facecolor='black', fancybox=True, framealpha=0.5)
    [text.set_color("white") for text in ax.legend.get_texts()] # colorindo cada um dos textos 
    # Inserindo título e sub-título e formatando
    plt.suptitle('Daily Returns Distribution Analysis',horizontalalignment='center', fontstyle= 'italic', fontsize=14, color='black')
    plt.title(f'Period: {init_date} - {final_date}',horizontalalignment='center', fontsize=12, color='darkblue')
    plt.show()

# -------------------- Volatilidade

def rolling_volatility_graph(strategy_returns: pd.Series, length:int=10, height:int=5, labelpad_size:int=10):
    '''
    Essa função plota o gráfico da volatilidade dos últimos 6 e 12 meses ao longo do tempo

    params: 
        strategy_returns: dataframe com retornos diários da estratégia
        length: comprimento do gráfico 
        height: altura do gráfico
    '''
    # Lidando com erros de input
    if not isinstance(strategy_returns, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')
    
    # Extraindo as datas finais e iniciais da análise para plotagem no gráfico
    init_date = strategy_returns.index.min().strftime('%d-%m-%Y')
    final_date = strategy_returns.index.max().strftime('%d-%m-%Y')
    
    # Criando as séries de volatilidade
    window1 = 21
    window3 = 21*3
    window6 = 21*6
    window12 = 21*12
    
    # Extraindo a volatilidade esperado
    vol_hist = stats.volatility(strategy_returns)
    # vol_hist = strategy_returns.std()*252**0.5
    
    # Criando a figura que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)
    
    # Plotando as diferenças janelas de corelação 
    vol_list = [] # lista para guardar as correlações
    if len(strategy_returns)>=window1:
        hist_std_1M = stats.rolling_volatility(strategy_returns,rolling_period=window1)
        # hist_std_1M = strategy_returns.rolling(window1).std()*252**0.5
        ax.plot(hist_std_1M, label='1 month', color='darkblue')
        # Armazenando a última observação para ser plotado na legenda
        vol_list.append(f'1M: {round(hist_std_1M[-1]*100,2)}%')
    if len(strategy_returns)>=window3:
        hist_std_3M = stats.rolling_volatility(strategy_returns,rolling_period=window3)
        # hist_std_3M = strategy_returns.rolling(window3).std()*252**0.5
        ax.plot(hist_std_3M, label='3 months', color='darkgreen')
        # Armazenando a última observação para ser plotado na legenda
        vol_list.append(f'3M: {round(hist_std_3M[-1]*100,2)}%')
    if len(strategy_returns)>=window6:
        hist_std_6M = stats.rolling_volatility(strategy_returns,rolling_period=window6)
        # hist_std_6M = strategy_returns.rolling(window6).std()*252**0.5
        ax.plot(hist_std_6M, label='6 months', color='purple')
        # Armazenando a última observação para ser plotado na legenda
        vol_list.append(f'6M: {round(hist_std_6M[-1]*100,2)}%')
    if len(strategy_returns)>=window12:
        hist_std_12M = stats.rolling_volatility(strategy_returns,rolling_period=window12)
        # hist_std_12M = strategy_returns.rolling(window12).std()*252**0.5
        ax.plot(hist_std_12M, label='6 months', color='maroon')
        # Armazenando a última observação para ser plotado na legenda
        vol_list.append(f'12M: {round(hist_std_12M[-1]*100,2)}%')
    
    # Armazenando a última observação para ser plotado na legenda
    vol_list.append(f'Historical: {round(vol_hist*100,2)}%')
    
    # Plotando a linha de sharpe histórico
    ax.axhline(y=vol_hist, label= 'Historical Volatility', color='crimson',linestyle='--')
    
    # Criando e formatando os labels do gráfico
    ax.set_xlabel(f'{" | ".join(vol_list)}', fontsize=13, color = 'darkblue', loc='right', labelpad=labelpad_size) # Add an x-label to the axes.
    ax.set_ylabel('Rolling Volatility (%)', fontsize=12, color = 'black') # Add a y-label to the axes.
    ax.tick_params(colors='black', which='both') # Colorindo os ticks dos eixos
    
    # Colorindo os eixos do gráfico
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # Colorindo o backgroun do gráfico
    ax.set_facecolor('lightgray')
    
    # Inseirndo grids
    ax.grid(True)
    # Ajustando o eixos y para porcentagens
    ax.yaxis.set_major_formatter(numfmt)
    
    # Inserindo a legenda no gráfico e colorindo o texto para branco
    ax.legend(loc='best', facecolor='black', fancybox=True, framealpha=0.5)
    [text.set_color("white") for text in ax.legend.get_texts()] # colorindo cada um dos textos 
    # Inserindo título e sub-título e formatando
    plt.suptitle('Rolling Volatility Analysis',horizontalalignment='center', fontstyle= 'italic', fontsize=14, color='black')
    plt.title(f'Period: {init_date} - {final_date}',horizontalalignment='center', fontsize=12, color='darkblue')
    plt.show()

# ----------------- Rolling Sharpe 12 meses

def rolling_sharpe_graph(strategy_returns:pd.Series, rf_returns:pd.Series=None, 
                         length=10, height=5, labelpad_size:int=10):
    # sourcery skip: low-code-quality
    '''
    Essa função plota o gráfico do sharpe dos últimos 12 meses ao longo do tempo

    params: 
        strategy_returns: dataframe com retornos diários da estratégia
        length: comprimento do gráfico 
        height: altura do gráfico
    '''
    # Lidando com erros de input
    if not isinstance(strategy_returns,pd.Series) and not isinstance(rf_returns,(pd.Series,type(None))):
        raise ValueError('ValueError: series dentro não é um data series.')
    
    # Extraindo as datas finais e iniciais da análise para plotagem no gráfico
    init_date = strategy_returns.index.min().strftime('%d-%m-%Y')
    final_date = strategy_returns.index.max().strftime('%d-%m-%Y')
    
    # Criando a quantidade de dias para as janelas de sharpe
    window1 = 21
    window3 = 21*3
    window6 = 21*6
    window12 = 21*12

    # Extraindo a volatilidade média do período
    if isinstance(rf_returns, type(None)):
        sharpe_hist = stats.sharpe(strategy_returns)
        # sharpe_hist = (strategy_returns.mean()/strategy_returns.std())*(252**0.5)
    else:
        
        sharpe_hist = stats.sharpe(strategy_returns, rf=rf_returns)
        # sharpe_hist = ((strategy_returns.mean()-rf_returns.mean())/strategy_returns.std())*(252**0.5)
    
    # Criando a figura que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)
    
    # Plotando as diferenças janelas de volatilidade 
    sharpe_list = [] # lista para guardar os rolling sharpes
    if len(strategy_returns)>=window1:
        # Caso a série tenha 1 mês ou mais em observações
        if isinstance(rf_returns, type(None)):
            # Desconsiderando o risk-free
            sharpe_1M = stats.rolling_sharpe(strategy_returns, rolling_period=window1)
            # sharpe_1M = (strategy_returns.rolling(window1).mean().div(strategy_returns.rolling(window1).std()))*(252**0.5)
        else:
            # Considerando o risk-free
            sharpe_1M = stats.rolling_sharpe(strategy_returns, rf=rf_returns, rolling_period=window1)
            # sharpe_1M = ((strategy_returns.rolling(window1).mean()-rf_returns.rolling(window1).mean()).div(strategy_returns.rolling(window1).std()))*(252**0.5)
        # Plotando a série no gráfico
        ax.plot(sharpe_1M, label='1 month', color='darkblue')
        # Armazenando a última observação para ser plotado na legenda
        sharpe_list.append(f'1M: {round(sharpe_1M[-1],2)}')
    if len(strategy_returns)>=window3:
        # Caso a série tenha 3 meses ou mais em observações
        if isinstance(rf_returns, type(None)):
            # Desconsiderando o risk-free
            sharpe_3M = stats.rolling_sharpe(strategy_returns, rolling_period=window3)
            # sharpe_3M = (strategy_returns.rolling(window3).mean().div(strategy_returns.rolling(window3).std()))*(252**0.5)
        else:
            # Considerando o risk-free
            sharpe_3M = stats.rolling_sharpe(strategy_returns, rf=rf_returns, rolling_period=window3)
            # sharpe_3M = ((strategy_returns.rolling(window3).mean()-rf_returns.rolling(window3).mean()).div(strategy_returns.rolling(window3).std()))*(252**0.5)
        # Plotando a série no gráfico
        ax.plot(sharpe_3M, label='3 months', color='darkgreen')
        # Armazenando a última observação para ser plotado na legenda
        sharpe_list.append(f'3M: {round(sharpe_3M[-1],2)}')
    if len(strategy_returns)>=window6:
        # Caso a série tenha 6 meses ou mais em observações
        if isinstance(rf_returns, type(None)):
            # Desconsiderando o risk-free
            sharpe_6M = stats.rolling_sharpe(strategy_returns, rolling_period=window6)
            # sharpe_6M = (strategy_returns.rolling(window6).mean().div(strategy_returns.rolling(window6).std()))*(252**0.5)
        else:
             # Considerando o risk-free
            sharpe_6M = stats.rolling_sharpe(strategy_returns, rf=rf_returns, rolling_period=window6)
            # sharpe_6M = ((strategy_returns.rolling(window6).mean()-rf_returns.rolling(window6).mean()).div(strategy_returns.rolling(window6).std()))*(252**0.5)
        # Plotando a série no gráfico
        ax.plot(sharpe_6M, label='6 months', color='maroon')
        # Armazenando a última observação para ser plotado na legenda
        sharpe_list.append(f'6M: {round(sharpe_6M[-1],2)}')
    if len(strategy_returns)>=window12:
        # Caso a série tenha 12 meses ou mais em observações
        if isinstance(rf_returns, type(None)):
            # Desconsiderando o risk-free
            sharpe_12 = stats.rolling_sharpe(strategy_returns, rolling_period=window12)
            # sharpe_12 = (strategy_returns.rolling(window12).mean().div(strategy_returns.rolling(window12).std()))*(252**0.5)
        else:
            # Considerando o risk-free
            sharpe_12 = stats.rolling_sharpe(strategy_returns, rf=rf_returns, rolling_period=window12)
            # sharpe_12 = ((strategy_returns.rolling(window12).mean()-rf_returns.rolling(window12).mean()).div(strategy_returns.rolling(window12).std()))*(252**0.5)
        # Plotando a série no gráfico
        ax.plot(sharpe_12, label='12 months', color='plum')
        # Armazenando a última observação para ser plotado na legenda
        sharpe_list.append(f'12M: {round(sharpe_12[-1],2)}')
    
    # Armazenando a média histórica  para ser plotado na legenda
    sharpe_list.append(f'Historical: {round(sharpe_hist,2)}')
    # Plotando a linha do eixo x (Y=0)
    ax.axhline(y=0, color='black')
    # Plotando a linha de sharpe histórico
    ax.axhline(y=sharpe_hist, label= 'Historical Sharpe', color='crimson',linestyle='--')
    
    # Criando e formatando os labels do gráfico
    ax.set_xlabel(f'{" | ".join(sharpe_list)}', fontsize=13, color = 'darkblue', loc='right', labelpad=labelpad_size) # Add an x-label to the axes.
    ax.set_ylabel('Rolling Sharpe Annualized', fontsize=12, color = 'black') # Add a y-label to the axes.
    ax.tick_params(colors='black', which='both') # Colorindo os ticks dos eixos
    
    # Colorindo os eixos do gráfico
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    # Colorindo o backgroun do gráfico
    ax.set_facecolor('lightgray')
    # Inseirndo grids
    ax.grid(True)
    
    # Inserindo a legenda no gráfico e colorindo o texto para branco
    ax.legend(loc='best', facecolor='black', fancybox=True, framealpha=0.5)
    [text.set_color("white") for text in ax.legend.get_texts()] # colorindo cada um dos textos 
    # Inserindo título e sub-título e formatando
    plt.suptitle('Rolling Sharpe Analysis',horizontalalignment='center', fontstyle= 'italic', fontsize=labelpad_size, color='black')
    plt.title(f'Period: {init_date} - {final_date}',horizontalalignment='center', fontsize=12, color='darkblue')
    plt.show()
    plt.close()

# ---------------- Rolling correlation

def rolling_correlation_graph(strategy_returns:pd.Series, bench_returns:pd.Series,
                              length=10, height=5, labelpad_size:int=10):
    '''
    Essa função plota o gráfico da correlação com outra série dos últimos 12 meses ao longo do tempo

    params: 
        strategy_returns: dataframe com retornos diários da estratégia
        other_returns: dataframe com retornos diários da outra estratégia
        length: comprimento do gráfico 
        height: altura do gráfico 
    '''
    if not isinstance(strategy_returns, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')
    
    # Extraindo as datas finais e iniciais da análise para plotagem no gráfico
    init_date = strategy_returns.index.min().strftime('%d-%m-%Y')
    final_date = strategy_returns.index.max().strftime('%d-%m-%Y')
    
    # Criando as séries de volatilidade
    window1 = 21
    window3 = 21*3
    window6 = 21*6
    window12 = 21*12
    
    # Extraindo a volatilidade histórica
    corr_hist = strategy_returns.corr(bench_returns)
    
    # Criando a figura que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)
    
    # Plotando as diferenças janelas de corelação 
    corr_list = [] # lista para guardar as correlações
    if len(strategy_returns)>=window1:
        # Caso a série tenha 1 mes ou mais em observações
        corr_1M = strategy_returns.rolling(window1).corr(bench_returns)
        # Plotando a série no gráfico
        ax.plot(corr_1M, label='1 month', color='darkblue')
        # Armazenando a última observação para ser plotado na legenda
        corr_list.append(f'1M: {round(corr_1M[-1],2)}')
    if len(strategy_returns)>=window3:
        # Caso a série tenha 3 meses ou mais em observações
        corr_3M = strategy_returns.rolling(window3).corr(bench_returns)
        # Plotando a série no gráfico
        ax.plot(corr_3M, label='3 months', color='darkgreen')
        # Armazenando a última observação para ser plotado na legenda
        corr_list.append(f'3M: {round(corr_3M[-1],2)}')
    if len(strategy_returns)>=window6:
        # Caso a série tenha 6 meses ou mais em observações
        corr_6M = strategy_returns.rolling(window6).corr(bench_returns)
        # Plotando a série no gráfico
        ax.plot(corr_6M, label='6 months', color='maroon')
        # Armazenando a última observação para ser plotado na legenda
        corr_list.append(f'6M: {round(corr_6M[-1],2)}')
    if len(strategy_returns)>=window12:
        # Caso a série tenha 12 meses ou mais em observações
        corr_12M = strategy_returns.rolling(window12).corr(bench_returns)
        # Plotando a série no gráfico
        ax.plot(corr_12M, label='12 months', color='plum')
        # Armazenando a última observação para ser plotado na legenda
        corr_list.append(f'12M: {round(corr_12M[-1],2)}')
    
    # Armazenando a última observação para ser plotado na legenda
    corr_list.append(f'Historical: {round(corr_hist,2)}')
    
    # Plotando a linha do eixo x (Y=0)
    ax.axhline(y=0, color='black')
    # Plotando a linha de sharpe histórico
    ax.axhline(y=corr_hist, label= 'Historical Correlation', color='crimson',linestyle='--')
    
    # Criando e formatando os labels do gráfico
    ax.set_xlabel(f'{" | ".join(corr_list)}', fontsize=13, color = 'darkblue', loc='right', labelpad=labelpad_size) # Add an x-label to the axes.
    ax.set_ylabel('Rolling Correlation', fontsize=12, color = 'black') # Add a y-label to the axes.
    ax.tick_params(colors='black', which='both') # Colorindo os ticks dos eixos
    
    # Colorindo os eixos do gráfico
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # Colorindo o backgroun do gráfico
    ax.set_facecolor('lightgray')
    # Inseirndo grids
    ax.grid(True)
    
    # Inserindo a legenda no gráfico e colorindo o texto para branco
    ax.legend(loc='best', facecolor='black', fancybox=True, framealpha=0.5)
    [text.set_color("white") for text in ax.legend.get_texts()] # colorindo cada um dos textos 
    # Inserindo título e sub-título e formatando
    plt.suptitle('Rolling Correlation Analysis',horizontalalignment='center', fontstyle= 'italic', fontsize=14, color='black')
    plt.title(f'Period: {init_date} - {final_date}',horizontalalignment='center', fontsize=12, color='darkblue')
    plt.show()
    plt.close()

# # -------------------------------------- TABELA DE DESEMPNHO 

def longest_drawdowns_graph(strategy_dict:dict, benchmark_dict:dict=None, worst_periods:int=5, 
                            length:int=10, height:int=5, compounded=True):
    """
    Função responsável por plotar a performance histórica da série
    deixando em evidência os piores "n" drawdowns.
    
    obs.: possibilidade de plotar também benchmarks para comparação.
    """
    # Extraindo as informações da série principal
    strategy_name = strategy_dict.get(0).get('name')
    strategy_color = strategy_dict.get(0).get('color')
    strategy_ret = strategy_dict.get(0).get('series')
    
    # Previnindo ValueErros 
    if not isinstance(strategy_ret, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')

    # Extraindo as datas finais e iniciais da análise para legenda do gráfico
    init_date = strategy_ret.index.min().strftime('%d-%m-%Y')
    final_date = strategy_ret.index.max().strftime('%d-%m-%Y')
    
    # Criando uma lista dos drawdowns das séries e escolhendo os "n" piores que serão evidenciados
    dddf = metrics.drawdown_details(strategy_ret)
    longest_dd = dddf.sort_values(by='days', ascending=False, kind='mergesort')[:worst_periods]

    # Criando a imagem que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)

    # Criando a série de retornos compostos 
    strategy_perf = utils.compsum(strategy_ret) if compounded else utils.cumsum(strategy_ret)
    # PLotando a série de retornos de retornos compostos
    ax.plot(strategy_perf, linewidth=2, label=strategy_name, color=strategy_color)
    
    # Checando se a lista com os dicts de benchmarks foi passada como input
    if not isinstance(benchmark_dict, type(None)) and isinstance(benchmark_dict, dict):
        # Caso passarmos os dicts de benchmarks
        for i in benchmark_dict:
            # Extraindo as informações das séries de benchmarks
            bench_name = benchmark_dict.get(i).get('name')
            bench_color = benchmark_dict.get(i).get('color')
            bench_ret = benchmark_dict.get(i).get('series')
            # Criando a série de retornos compostos
            bench_perf = utils.compsum(bench_ret) if compounded else utils.cumsum(bench_ret)
            # Só plataremos o benchmark se for uma serie
            if isinstance(bench_perf, pd.Series):
                # Plotando a série de benchmark
                ax.plot(bench_perf, label=bench_name, color=bench_color)
                
    # Evidenciando os "n" piores drawdowns das séries em vermelho no gráfico
    for _, row in longest_dd.iterrows():
        ax.axvspan(*mdates.datestr2num([str(row['start']), str(row['end'])]),
                   color='red', alpha=.2)

    # Plotando a linha do eixo x (Y=0)
    ax.axhline(y=0, color='black')
    
    # Criando e formatando os labels do gráfico
    ax.set_ylabel('Historical Performance (%)', fontsize=12, color = 'black') # Add a y-label to the axes.
    ax.tick_params(colors='black', which='both') # Colorindo os ticks dos eixos
    
    # Colorindo os eixos do gráfico
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # Colorindo o backgroun do gráfico
    ax.set_facecolor('lightgray')
    # Inseirndo grids
    ax.grid(True)
    # Ajustando o eixos y para porcentagens
    ax.yaxis.set_major_formatter(numfmt)
    
    # Inserindo a legenda no gráfico e colorindo o texto para branco
    ax.legend(loc='best', facecolor='black', fancybox=True, framealpha=0.5)
    [text.set_color("white") for text in ax.legend.get_texts()] # colorindo cada um dos textos 

    # Inserindo título e sub-título e formatando
    plt.suptitle('5 Worst Drawdowns Analysis',horizontalalignment='center', fontstyle= 'italic', fontsize=14, color='black')
    plt.title(f'Period: {init_date} - {final_date}',horizontalalignment='center', fontsize=12, color='darkblue')
    plt.show()
    plt.close()
    
def monthly_heatmap_graph(strategy_ret:pd.Series, annot_size:int=10,grayscale:bool=False,
                          cbar:bool=True, square:bool=False, compounded:bool=True, 
                          eoy:bool=False, length:int=10, height:int=5):
    """
    Função responsável pela geração de um heatmap de retornos mensais da séries de 
    retornos de interesse.
    """
    # Previnindo ValueErros 
    if not isinstance(strategy_ret, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')
    
    # Criando os retornos agregados por "eoy"   
    returns = metrics.monthly_returns(strategy_ret, eoy=eoy,
                                     compounded=compounded) * 100
    
    # Criando a imagem que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout() 
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)
    
    # Removendo os eixos da imagem
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Setando as cores do gráfico
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    # Setando a escala de cores desejada para o heatmap
    cmap = 'gray' if grayscale else 'RdYlGn'
     
    # Aplicando o heatmap no gráfico de retornos
    ax = sns.heatmap(returns, ax=ax, annot=True, center=0,
                      annot_kws={"size": annot_size},
                      fmt="0.2f", linewidths=0.5,
                      square=square, cbar=cbar, cmap=cmap,
                      cbar_kws={'format': '%.0f%%'})

    # Setando o título do gráfico
    ax.set_title('    Monthly Returns Analysis (%)\n',fontstyle= 'italic', fontsize=14, color='black')
    
    # Criando e formatando os labels do gráfico
    ax.set_ylabel('Years', color = 'black', fontsize=12)
    # Ajustando as coordenadas do gráfico
    ax.yaxis.set_label_coords(-.1, .5)
    ax.tick_params(colors="#808080")
    plt.xticks(rotation=0, fontsize=annot_size*1.2)
    plt.yticks(rotation=0, fontsize=annot_size*1.2)
    plt.subplots_adjust(hspace=0, bottom=0, top=1)
    fig.tight_layout(w_pad=0, h_pad=0)
    
    # Plotando a figura
    plt.show()
    plt.close() # Fechando a figura


def distribution_graph(strategy_ret:pd.Series, grayscale=False, compounded=True,
                       length:int=10, height:int=5):

    # Previnindo ValueErros 
    if not isinstance(strategy_ret, pd.Series):
        raise ValueError('ValueError: series dentro do dict strategy não é um data series.')

    # Extraindo as datas finais e iniciais da análise para legenda do gráfico
    init_date = strategy_ret.index.min().strftime('%d-%m-%Y')
    final_date = strategy_ret.index.max().strftime('%d-%m-%Y')

    # Cores que utilizaremos nos boxplots 
    colors = ["#fedd78", "#348dc1", "#af4b64",
              "#4fa487", "#9b59b6", "#808080"]

    # Caso passarmos grayscale
    if grayscale:
        colors = ['#f9f9f9', '#dddddd', '#bbbbbb', '#999999', '#808080']

    # Criando um df para armazenar os diferentes tipos de amostras (por periodicidade)
    port = pd.DataFrame(strategy_ret.fillna(0))
    port.columns = ['Daily']

    # Definindo como iremos sumarirar os retornos por período 
    apply_fnc = utils.comp if compounded else utils.sum

    # Criando as amostras e preenchendo o df
    port['Weekly'] = port['Daily'].resample('W-MON').apply(apply_fnc)
    port['Weekly'].ffill(inplace=True)

    port['Monthly'] = port['Daily'].resample('M').apply(apply_fnc)
    port['Monthly'].ffill(inplace=True)

    port['Quarterly'] = port['Daily'].resample('Q').apply(apply_fnc)
    port['Quarterly'].ffill(inplace=True)

    port['Yearly'] = port['Daily'].resample('A').apply(apply_fnc)
    port['Yearly'].ffill(inplace=True)

    # Criando a imagem que dará forma ao gráfico
    fig = plt.figure(figsize=(length, height),facecolor='white')
    fig.tight_layout()
    # Inserindo o gráfico dentro da imagem
    ax = fig.add_subplot(111)

    # Excluindo os eixos do gráfico 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Inserindo título e sub-título e formatando
    plt.suptitle('     5 Worst Drawdowns Analysis',fontstyle= 'italic', 
                 fontsize=14, color='black')
    plt.title(f'Period: {init_date} - {final_date}', fontsize=12, color='darkblue')

    # Setando a cor de fundo do gráfico 
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    # Plotando os boxplots
    sns.boxplot(data=port, ax=ax, palette=tuple(colors[:5]))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, loc: "{:,}%".format(int(x*100))))

    # Ajustando as coordenadas
    ax.yaxis.set_label_coords(-.1, .5)
    fig.autofmt_xdate()

    # Ajsutando o espçamento entre os plots
    with contextlib.suppress(Exception):
        plt.subplots_adjust(hspace=0)
    # Ajustando o layout do gráfico
    with contextlib.suppress(Exception):
        fig.tight_layout(w_pad=0, h_pad=0)
    plt.show()
    plt.close()

