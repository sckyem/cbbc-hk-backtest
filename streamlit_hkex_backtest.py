from my_io import read_csv, read_parquet
from my_mongodb import Mongodb
import streamlit as st
from default_modules import *
import itertools
from my_backtests import qcut_mask, sma_mask, pnl, ect_cpnl, performance
from collections import OrderedDict

root = 'hkex'
coffee = "https://buymeacoffee.com/sckyem"
element_names = [  'Underlyings', 'Data Name', 'Market', 'MCE', 'Aggregate'  ]
neglected_symbols = [  'HSTEC'  ]

def next(): 
    st.session_state.page += 1
def back(): 
    st.session_state.page -= 1
def restart(): 
    st.session_state.page = 0

def yfinance_symbol(symbols):
    def func(symbol):
        match symbol:
            case 'HSTEC':
                return 'HSTECH.HK'
            case 'HSCEI':
                return '^HSCE'
        if str(symbol).isalpha():
            return f'^{symbol}'
        elif str(symbol).isnumeric() and len(str(symbol)) > 4:
            return f'{symbol[-4:]}.HK'
        else:
            return symbol

    if isinstance(symbols, list):
        return [  func(i) for i in symbols  ]
    else:
        return func(symbols)
    
@st.cache_data(ttl=28800)
def load_from(source, collection='test', document='test', query={}, projection={}):
    match source:
        case "Parquet":
            df = read_parquet(root, 'cbbc', 'cbbc')
        case "CSV":
            df = read_csv(root, 'cbbc', 'cbbc')
        case "MongoDB":
            document = Mongodb(collection=collection, document=document)
            df = document.read(query, projection, is_dataframe=True)            
    return df

def element(list_of_comma_separated_strings):
    return [  sorted(list(set(t))) for t in zip(*[  str(i).split(',') for i in list_of_comma_separated_strings  ])  ]

def filter(symbols_benchmarks, classifications, fee):

    min_annualized_sr = st.sidebar.number_input(  "min_annualized_sr", value=1.2, step=0.1  )
    min_num_trades = st.sidebar.number_input(  "min_num_trades", 0, value=6, step=1  )
    min_avg_pnl = st.sidebar.number_input(  "min_avg_pnl", value=0.0, step=0.01  )
    min_avg_pnl_to_mdd = st.sidebar.number_input(  "min_avg_pnl_to_mdd", 0.0, value=0.2, step=0.1  )
    max_mdd = st.sidebar.number_input(  "max_mdd", value=0.2, step=0.05  )
    better_than_benchmark = st.sidebar.number_input(  "better_than_benchmark", value=0.1, step=0.05  )

    if min_num_trades:
        classifications = classifications.loc[:, classifications.apply(  lambda x: (x.shift(1) == 0) & (x != 0)  ).sum() >= min_num_trades]

    # starts = classifications.apply(lambda x: (x.shift(1) == 0) & (x != 0)).astype(int)    
    if classifications.eq(1).all().all() or classifications.empty:
        pnls = []
        for symbol, benchmarks in symbols_benchmarks.items():
            for b in benchmarks:
                benchmark = benchmarks[b]
                if b in ['Close_Open', 'Intraday']:
                    benchmark -= fee*2
                pnls.append(benchmark.rename(  f"{symbol},{b}"  ))            
        pnls = pd.concat(pnls, axis=1)
        pnls.iloc[0] = -fee
        classifications = pd.DataFrame(1, pnls.index, pnls.columns)
    else:
        pnls = [  pnl(v, classifications.filter(like=f"{k}", axis=1), fee, better_than_benchmark) for k, v in symbols_benchmarks.items()  ]
        if all(x is None for x in pnls):
            pnls = None
        else:
            pnls = pd.concat(pnls, axis=1)
    
    if pnls is None:
        performances = None
    else:
        performances = performance(pnls, classifications)
        # Filter
        if min_avg_pnl_to_mdd:
            performances = performances[performances['avg_pnl_to_mdd'] >= min_avg_pnl_to_mdd]
        if min_annualized_sr:
            performances = performances[performances['annualized_sr'] >= min_annualized_sr]
        if min_avg_pnl:
            performances = performances[performances['avg_pnl'] >= min_avg_pnl]
        if max_mdd:
            performances = performances[performances['mdd'] <= max_mdd]
        chart_order = st.sidebar.selectbox(  "chart_order", performances.columns.to_list(), len(performances.columns.to_list())-1  )
        performances = performances.sort_values(  chart_order, ascending=True if chart_order in [  'mdd', 'annualized_std'  ] else False  )   
        
        pnls = pnls[performances.index]

        num = len(str(classifications.columns[0]).split(','))
        combinations = [  str(c).split(',')[:num] for c in performances.index  ]
        
        combinations = list(OrderedDict.fromkeys([  ','.join(c) for c in combinations  ]))
        combinations = [  c for c in combinations if c in classifications.columns  ]
        
        if combinations:
            classifications = classifications[combinations]
    return classifications, pnls, performances

def app():

    st.set_page_config(
        layout='wide',
        page_title="Historical Data of CBBC", 
        page_icon="📈",
        menu_items={            
            'About': "Created by CKY"
            }
        )

    st.sidebar.title("Historical Data of CBBC", anchor=False)

    df = load_from("MongoDB", 'cbbc', 'cbbc').dropna(how='all')
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = columns_to_strings(df.columns)

    from_time = st.sidebar.radio(  "Date Range", ["3M", "1Y", "All"], 1, horizontal=True  )            
    if from_time != 'All':
        df = df.loc[df.index[-2] - interval_to_timedelta(from_time):]
    
    elements = element(df.columns)
    # For local
    # elements_selected = [  st.sidebar.multiselect(f"{element_names[i]}", ['All'] + e, 'All') for i, e in enumerate(elements) ]
    # For Online
    elements_selected = [  st.sidebar.multiselect(f"{element_names[i]}", ['All'] + e if i > 0 else e, 'All' if i > 0 else 'HSI') for i, e in enumerate(elements) ]
    elements_selected = [  elements[i] if 'All' in e else e for i, e in enumerate(elements_selected)  ]
    elements_selected = [  ','.join(list(i)) for i in itertools.product(*elements_selected)  ]
    
    if elements_selected:
        df = df.loc[:, df.columns.isin(elements_selected)]

    charts_per_tab = st.sidebar.select_slider(  "No of Charts per Tab", list(range(1, 101)), 10  )
    height = st.sidebar.select_slider(  "chart_height", list(range(200, 1001, 50)), 350  )

    chart_selected = st.sidebar.selectbox(  "chart_selected", ["Independent", "Classification", "Start", "Pnl", "Cpnl", "TA"], 4  )
    ta_selected = ''
    ta_window_selected = 0
    ta_std_selected = 0
    if 'TA' in chart_selected:
        ta_optoins = ["", "SMA", "BB"]
        ta_selected = st.sidebar.selectbox(  "ta_selected", ta_optoins, 0  )
        if ta_selected:
            ta_window_selected = st.sidebar.select_slider(  "windows", list(range(5, 51, 5)), 5  )
            if ta_selected == 'BB':
                ta_std_selected = st.sidebar.select_slider(  "std", [  i/10 for i in list(range(0, 51, 5))  ], 2.0  )

    chart_type_selected = st.sidebar.selectbox(  "Chart Type", ["Line", "Table", "None"], 0  )

    statistic_selected = st.sidebar.selectbox(  "Independent Statistic ", ["", "Diff", "Diff_Diff", "Pct"], 0  )
    classification_selected = st.sidebar.selectbox(  "Independent Classification", ["", "Quantile"], 1  )

    match classification_selected:
        case "Quantile":
            num_groups = st.sidebar.selectbox(  "Bins", [2, 4, 5, 8, 10, 20], 0  )
            rolling_window = 0 # st.sidebar.slider(  "rolling_window", 1, 20, 1, 1  )

    pnl_optoins = ["Open_Open", "Close_Close", "Close_Open", "Intraday"]
    ohlcv_optoins = ["Open", "High", "Low", "Close", "Volume"]
    appendix_selected = st.sidebar.selectbox(  "Appendix", [""] + pnl_optoins + ohlcv_optoins, 1  )
    
    last_nday = st.sidebar.number_input("last_nday", -1, value=1, step=1)

    fee = st.sidebar.number_input(  "Fee (%)", 0.0, 2.0, 1.0, 0.1  )
    fee = fee / 100

    if df is not None and not df.empty:
        
        start = df.index[0]
        end = df.index[-1]
        start_time_delta = st.sidebar.number_input("start_time_delta", -1, 21, 0, 1)
        duration = 1 # st.sidebar.number_input("duration", 1, 21, 1, 1)

        symbols = list(set(  [str(i).split(',')[0] for i in df.columns]  ))    
        symbols = [  i for i in symbols if i not in neglected_symbols  ]
        symbols_ohlcvs = {  i:load_from("MongoDB", 'yfinance', yfinance_symbol(i), {'_id': {'$gte': start, '$lte': end}}) for i in symbols  }
        symbols_benchmarks = {  k:get_log_returns(v, start_time_delta, duration) for k,v in symbols_ohlcvs.items() if v is not None  }

        match statistic_selected:
            case "Diff": 
                independents = df.diff().dropna()
            case "Diff_Diff":
                independents = df.diff().diff().dropna()
            case "Pct": 
                independents = df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
            case _:
                independents = df
        
        match classification_selected:
            case "Quantile":
                classifications = qcut_mask(  independents, num_groups, rolling_window, last_nday  )
            case _:
                classifications = pd.DataFrame(1, index=independents.index, columns=independents.columns  )
        
        classifications, pnls, performances = filter(symbols_benchmarks, classifications, fee)

        cpnls = None
        ta_pnls = None
        if pnls is not None:
            cpnls = pnls.cumsum()

            if ta_selected and ta_window_selected:
                ta_pnls = ect_cpnl(pnls, cpnls, fee, ta_selected, ta_window_selected, ta_std_selected)

        msgs = []
        if independents is not None:
            msgs.append(  f"Independents: {independents.index[-1].strftime('%Y-%m-%d (%a)')}"  )
        if pnls is not None:
             msgs.append(  f"Pnl: {pnls.index[-1].strftime('%Y-%m-%d (%a)')}"  )
        st.write(  ','.join(msgs)  )

        if performances is not None:
            summary = pd.DataFrame([str(i).split(',') for i in performances.index.tolist()])
            if len(summary.columns) > 2:
                summary = summary[[summary.columns[0], summary.columns[-2], summary.columns[-1]]]
                summary.columns = ['underlying', 'order time', 'position']
                summary['position'] = summary['position'].map(lambda x: 1 if x == 'long' else -1 if x == 'short' else 0)
                summary['order time'] = summary['order time'].map(lambda x: 'Open' if x == 'Open_Open' else 'Close' if x == 'Close_Close' else np.nan)
                summary = pd.pivot_table(summary, index='order time', columns='underlying', values='position', aggfunc='sum', fill_value=0)
                summary['Position'] = summary.sum(axis=1)
                summary = summary.sort_index(ascending=False)
                summary.loc['Position'] = summary.sum()
                st.dataframe(  summary, use_container_width=True  )

            st.dataframe(  performances.mean().rename('mean').to_frame().T, use_container_width=True  )
            st.dataframe(  performances.style.highlight_max(axis=0), use_container_width=True  )

        match chart_selected:
            case 'Independent':
                charts = independents
            case 'Classification':
                charts = classifications
            case 'Start':
                pass # charts = starts
            case 'Pnl':
                charts = pnls
            case 'Cpnl':
                charts = cpnls
            case 'TA':
                charts = ta_pnls

        if charts is not None and not charts.empty:
            
            match chart_type_selected:
                case 'Line':
                    tab_names = [  str(i+1) for i in range(0, math.ceil(len(charts.columns) / charts_per_tab))]
                    for i, tab in enumerate(st.tabs(tab_names)):
                    
                        with tab:
                            cols = charts.columns[  i*charts_per_tab:(i+1)*charts_per_tab  ].to_list()
                            if cols:
                                for j in cols:

                                    st.write(j)
                                    chart = charts[j].dropna()
                                    name = f"{appendix_selected},Appendix"
                                    symbol = str(j).split(',')[0]
                                    appendix = None
  
                                    if appendix_selected in ohlcv_optoins:
                                        ohlcv = symbols_ohlcvs.get(symbol)
                                        if ohlcv is not None:
                                            appendix = ohlcv[appendix_selected].rename(name)
                                            if appendix is not None:                                    
                                                appendix = get_scaled_df(appendix, chart.min(), chart.max())

                                    elif appendix_selected in pnl_optoins:
                                        benchmark = symbols_benchmarks.get(symbol)
                                        if benchmark is not None:
                                            appendix = benchmark[appendix_selected].cumsum().rename(name)                                            
                                            if appendix is not None:
                                                appendix.iloc[0] = 0
                                                if chart_selected in ['Independent', 'Classification', 'Start']:
                                                    appendix = get_scaled_df(appendix, chart.min(), chart.max())

                                    if chart_selected == 'TA':
                                        x = ','.join(str(j).split(',')[:-1])
                                        cpnl = cpnls[x]
                                        cpnl = cpnl.loc[cpnls.index.isin(chart.index)]
                                        chart = pd.concat([chart, cpnl], axis=1).ffill()
                                        chart.iloc[0] = 0                                     

                                    if appendix is not None:
                                        appendix = appendix.loc[appendix.index.isin(chart.index)]
                                        st.line_chart(  pd.concat([chart, appendix], axis=1), height=height  )     
                                    else:
                                        st.line_chart(  chart, height=height  )                                           

                case 'Table':
                   st.dataframe(charts, use_container_width=True)
                                
        

    # if st.session_state.page == 1:

    #     df = st.session_state.cbbc
    #     st.title("Compare", anchor=False)

    #     compare = st.sidebar.selectbox("Compare", ["Proportion", "Ratio"], 1)
    #     if compare == 'Ratio':
    #         is_logscale = st.sidebar.toggle("Log Scale", False)

    #     is_inverse = st.sidebar.toggle("Inverse", False)        

    #     if compare:
    #         st.sidebar.subheader("No of Elements per Combination:")
    #         combinations_limit = [  st.sidebar.select_slider(f"{element_names[i]}", [1, 2], 1) if len(set) > 1 else 1 for i, set in enumerate(element(df.columns))  ]
    #         if all([i == 1 for i in combinations_limit]):
    #             st.write("No combination")
    #         else:
    #             combinations = [  c for c in combination(df.columns, max(combinations_limit)) if all(a <= b for a, b in zip([len(set(t)) for t in zip(*[  str(i).split(',') for i in c])], combinations_limit))  ]
                
    #             tab_names = [  str(i+1) for i in range(0, math.ceil(len(combinations) / lines_per_tab))  ]
    #             for i, tab in enumerate(st.tabs(tab_names)):
                    
    #                 symbols = list(set(  [str(i).split(',')[0] for i in df.columns]  ))
    #                 symbols_closes = {  i:load_from("MongoDB", 'yfinance', yfinance_symbol(i), {'_id': {'$gte': df.index[0], '$lte': df.index[-1]}}, {'_id':1, 'Close':1}) for i in symbols  }

    #                 with tab:
    #                     tab_cols = combinations[  i*lines_per_tab:i*lines_per_tab+lines_per_tab  ]
    #                     for j in range(  0, lines_per_tab, lines_per_chart  ):
    #                         chart_cols = tab_cols[  j:j+lines_per_chart  ]

    #                         if chart_cols:      
    #                             symbol = str(chart_cols[0][0]).split(',')[0]
    #                             close = symbols_closes[symbol]

    #                             if is_inverse:                                    
    #                                 a = chart_cols[0][1]
    #                                 b = chart_cols[0][0]
    #                             else:                                    
    #                                 a = chart_cols[0][0]
    #                                 b = chart_cols[0][1]
                                    
    #                             match compare:
    #                                 case "Proportion":
    #                                     chart_df = (df[a] / (df[a] + df[b])).rename('Proportion')
    #                                     st.write(  f"{a} / ( {a} + {b} )"  )
    #                                 case "Ratio":
    #                                     chart_df = (df[a] / df[b]).rename('Ratio')
    #                                     if is_logscale:
    #                                         chart_df = np.log(chart_df)
    #                                     chart_df = chart_df.replace([np.inf, -np.inf], np.nan)                                            
    #                                     st.write(  f"{a} / {b}"  )

    #                             if is_show_charts:
    #                                 if is_show_close:
    #                                     if close is not None:
    #                                         close = get_scaled_df(close, chart_df.min().min(), chart_df.max().max())
    #                                         chart_df = pd.concat([close, chart_df], axis=1)                      
    #                                 tab.line_chart(  chart_df, height=chart_height  )
    #                             else:
    #                                 if is_show_close:
    #                                     chart_df = pd.concat([close, chart_df], axis=1)
    #                                 tab.dataframe(  chart_df  )


    # if st.session_state.page != 0:
    #     st.sidebar.button("Back", on_click=back)
    # st.sidebar.button("Next", on_click=next)
    # st.sidebar.button("Restart", on_click=restart)
    # st.sidebar.link_button("☕ Buy Me a Coffee", coffee)

if __name__ == '__main__':

    app()