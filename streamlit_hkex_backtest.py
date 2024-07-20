from my_io import read_csv, read_parquet
from my_mongodb import Mongodb
import streamlit as st
from default_modules import *
import itertools
from my_backtests import qcut_mask, sign_mask, pnl_charge, performance

root = 'hkex'
coffee = "https://buymeacoffee.com/sckyem"
element_names = [  'Symbol', 'Data Name', 'Market', 'MCE', 'Aggregate'  ]
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
            client = Mongodb(collection=collection, document=document)
            df = client.read(query, projection, is_dataframe=True)
    if df is not None:
        df.columns = columns_to_strings(df.columns)
        return df

def filter(symbols_benchmarks={}, symbols_masks={}, fee=0):
    
    min_annualized_pnl = st.sidebar.number_input(  "min_annualized_pnl", value=0.2, step=0.1  )
    min_annualized_sr = st.sidebar.number_input(  "min_annualized_sr", value=1.2, step=0.1  )
    min_avg_pnl = st.sidebar.number_input(  "min_avg_pnl", value=0.01, step=0.01  )
    max_mdd = st.sidebar.number_input(  "max_mdd", value=0.2, step=0.05  )
    min_num_trades = st.sidebar.number_input(  "min_num_trades", 0, value=6, step=1  )
    min_calmar = st.sidebar.number_input(  "min_calmar", 0.0, value=2.0, step=0.1  )
    better_than_benchmark = st.sidebar.number_input(  "better_than_benchmark", value=0.1, step=0.05  )

    symbols_num_trades = {  k:v.apply(  lambda x: (x.shift(1) == 0) & (x != 0)  ).sum() for k,v in symbols_masks.items() if v is not None  }
    
    if min_num_trades:
        symbols_masks = {  k:v.loc[:, symbols_num_trades.get(k) >= min_num_trades] for k,v in symbols_masks.items() if v is not None  }
    # starts = primary_masks.apply(lambda x: (x.shift(1) == 0) & (x != 0)).astype(int)
    
    symbols_pnls = {}
    symbols_charges = {}
    for k,v in symbols_masks.items():
        if not v.empty:
            benchmarks = symbols_benchmarks.get(k)
            pnls_charges = pnl_charge(benchmarks, v, fee, better_than_benchmark)
            if pnls_charges is not None:
                symbols_pnls.update(  {k:pnls_charges.get('pnls')}  )
                symbols_charges.update(  {k:pnls_charges.get('charges')}  )

    if symbols_pnls:
        symbols_exposures = {  k:v.sum() for k,v in symbols_masks.items() if v is not None  }
        symbols_performances = {  k:performance(v, symbols_num_trades.get(k), symbols_exposures.get(k)) for k,v in symbols_pnls.items()  }
        
        # Filter
        if min_calmar:
            symbols_performances = {  k:v[v['calmar'] >= min_calmar] for k,v in symbols_performances.items()  }
        if min_annualized_sr:
            symbols_performances = {  k:v[v['annualized_sr'] >= min_annualized_sr] for k,v in symbols_performances.items() if not v.empty  }
        if min_annualized_pnl:
            symbols_performances = {  k:v[v['annualized_pnl'] >= min_annualized_pnl] for k,v in symbols_performances.items() if not v.empty  }
        if min_avg_pnl:
            symbols_performances = {  k:v[v['avg_pnl'] >= min_avg_pnl] for k,v in symbols_performances.items() if not v.empty  }
        if max_mdd:
            symbols_performances = {  k:v[v['mdd'] <= max_mdd] for k,v in symbols_performances.items() if not v.empty  }
        symbols_performances = {  k:v for k,v in symbols_performances.items() if not v.empty  }
    else:
        symbols_performances = {}
    
    if symbols_performances:
        performances = pd.concat([  v for k,v in symbols_performances.items() if not v.empty  ])

        perf_options = performances.columns.to_list()
        chart_order = st.sidebar.selectbox(  "chart_order", perf_options, len(perf_options) -1  )
        performances = performances.sort_values(  chart_order, ascending=True if chart_order in [  'mdd', 'annualized_std'  ] else False  )   
        
        pnls = pd.concat([  v for k,v in symbols_pnls.items() if not v.empty  ], axis=1)
        pnls = pnls[performances.index]

        charges = pd.concat([  v for k,v in symbols_charges.items() if not v.empty  ], axis=1)        
        masks = pd.concat([  v for k,v in symbols_masks.items() if not v.empty  ], axis=1)
    else:
        masks, pnls, charges, performances = None, None, None, None

    return masks, pnls, charges, performances

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

    from_time = st.sidebar.radio(  "Date Range", ["3M", "1Y", "All"], 1, horizontal=True  )        

    elements = [  sorted(list(set(t))) for t in zip(*[  str(i).split(',') for i in df.columns  ])  ]
    symbols = elements[0]
    symbols = [  i for i in symbols if i not in neglected_symbols  ]
    elements[0] = symbols

    elements_selected = [  st.sidebar.multiselect(f"{element_names[i]}", e) for i, e in enumerate(elements) ]    
    elements_selected = [  e if e else elements[i] for i, e in enumerate(elements_selected)  ]
    columns_filtered = [  ','.join(list(i)) for i in itertools.product(*elements_selected)  ]

    statistic = st.sidebar.selectbox(  "statistic", ["", "Speed", "Accel", "Pct"], 0  )
    mask_selected = st.sidebar.selectbox(  "mask_selected", ["", "Quantile"], 1  )
    match mask_selected:
        case "Quantile":
            num_groups = st.sidebar.selectbox(  "num_groups", [2, 4, 5, 8, 10, 20], 1  )
            rolling_window = 0 # st.sidebar.slider(  "rolling_window", 1, 20, 1, 1  )

    statistic_2nd = st.sidebar.selectbox(  "statistic_2nd", ["", "Close_Open", "Intraday"], 0  )
    mask_2nd_selected = st.sidebar.selectbox(  "mask_2nd_selected", ["", "Sign"], 1  )

    start_time_delta = st.sidebar.number_input("start_time_delta", 0, 21, 1, 1)
    duration = 1 # st.sidebar.number_input("duration", 1, 21, 1, 1)

    last_nday = st.sidebar.number_input(  "last_nday", -1, value=1, step=1  )
    fee = st.sidebar.number_input(  "Fee(%)", 0.0, 2.0, 1.0, 0.1  )
    fee = fee / 100

    pnl_optoins = ["Open_Open", "Close_Close", "Close_Open", "Intraday"]
    ohlcv_optoins = ["Open", "High", "Low", "Close", "Volume"]
    appendix_selected = st.sidebar.selectbox(  "Appendix", [""] + pnl_optoins + ohlcv_optoins, 1  )
    
    chart_type = st.sidebar.selectbox(  "Chart Type", ["Line", "Table", "None"], 0  )
    chart_selected = st.sidebar.selectbox(  "chart_selected", ["Independent", "masks_1st", "Dependents", "masks_2nd", "masks", "log_returns", "OHLCV", "Benchmark", "Pnl", "Charge", "Cpnl", "TA"], 10  )
    charts_per_tab = st.sidebar.select_slider(  "No of Charts per Tab", list(range(1, 101)), 10  )
    height = st.sidebar.select_slider(  "chart_height", list(range(200, 1001, 50)), 350  )

    if 'TA' in chart_selected:
        ta_optoins = ["", "SMA", "BB"]
        ect_selected = st.sidebar.selectbox(  "ect", ta_optoins, 0  )
        if ect_selected:
            ect_window = st.sidebar.select_slider(  "windows", list(range(5, 51, 5)), 5  )
            if ect_selected == 'BB':
                ect_std = st.sidebar.select_slider(  "std", [  i/10 for i in list(range(0, 51, 5))  ], 2.0  )
    else:        
        ect_selected = ''
        ect_window = 0
        ect_std = 0

    if df is not None:
            
        if from_time != 'All':
            df = df.loc[df.index[df.index.isin(pd.date_range(start=df.index[-2] - interval_to_timedelta(from_time), end=df.index[-1], freq='D'))]]            
        
        if columns_filtered:
            df = df.loc[:, df.columns.isin(columns_filtered)]

        if not df.empty:
            
            start = df.index[0]
            end = df.index[-1]
            symbols_dfs = {  i:df.filter(like=f"{i}", axis=1) for i in symbols  }

            match statistic:
                case "Diff": 
                    symbols_independents = {  k:v.diff() for k,v in symbols_dfs.items()  }
                case "Diff_Diff":
                    symbols_independents = {  k:v.diff().diff() for k,v in symbols_dfs.items()  }
                case "Pct": 
                    symbols_independents = {  k:v.pct_change().replace([np.inf, -np.inf], np.nan) for k,v in symbols_dfs.items()  }
                case _:
                    symbols_independents = symbols_dfs.copy()
                
            match mask_selected:
                case "Quantile":
                    symbols_masks_1st = {  k:qcut_mask(  v, num_groups, rolling_window, last_nday  ) for k,v in symbols_independents.items()  }
                case _:
                    symbols_masks_1st = {  k:pd.DataFrame(1, index=v.index, columns=v.columns  ) for k,v in symbols_independents.items()  }
        
        symbols_ohlcvs = {  i:load_from("MongoDB", 'yfinance', yfinance_symbol(i), {'_id': {'$gte': start, '$lte': end}}) for i in symbols  }
        symbols_log_returns = {  k:get_log_returns(v, -1, duration) for k,v in symbols_ohlcvs.items() if v is not None  }
        symbols_benchmarks = {  k:v.shift(  -start_time_delta  )[['Open_Open', 'Close_Close']] for k,v in symbols_log_returns.items() if v is not None  }
        
        match statistic_2nd:
            case 'Close_Open':
                symbols_dependents = {  k:v[['Close_Open']] for k,v in symbols_log_returns.items()  }
            case 'Intraday':
                symbols_dependents = {  k:v[['Intraday']] for k,v in symbols_log_returns.items()  }
            case _:
                symbols_dependents = None

        if symbols_dependents is None:
            symbols_masks_2nd = None
        else:
            match mask_2nd_selected:
                case 'Sign':
                    symbols_masks_2nd = {  k:sign_mask(v) for k,v in symbols_dependents.items()  }
        
        if symbols_masks_1st is not None and symbols_masks_2nd is not None:
            symbols_masks = {}
            for s in symbols:
                masks_1 = symbols_masks_1st.get(s)
                masks_2 = symbols_masks_2nd.get(s)
                if masks_1 is not None and masks_2 is not None:
                    masks = []
                    for i in masks_1:
                        for j in masks_2:
                            df = pd.concat([masks_1[i], masks_2[j]], axis=1)
                            df = (df == 1).all(axis=1).astype(int).rename(  f"{i},{j}"  )
                            masks.append(df)
                    masks = pd.concat(masks, axis=1)
                    symbols_masks.update({s:masks})

        elif symbols_masks_1st is not None:
            symbols_masks = symbols_masks_1st.copy()
        elif symbols_masks_2nd is not None:
            symbols_masks = symbols_masks_2nd.copy()
        else:
            symbols_masks = None
        masks, pnls, charges, performances = filter(symbols_benchmarks, symbols_masks, fee)

        if pnls is None:
            cpnls = None
            ects = None
        else:
            cpnls = pnls.cumsum()
    #         if ect_selected and ect_window:
    #             ects = ect_cpnl(pnls, cpnls, fee, ect, ect_window, ect_std)
        msgs = []
        if df is not None:
            msgs.append(  f"Independents: {df.index[-1].strftime('%Y-%m-%d (%a)')}"  )
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
                charts = pd.concat([  v for k, v in symbols_independents.items()  ], axis=1)
            case 'masks_1st':
                charts = pd.concat([  v for k, v in symbols_masks_1st.items()  ], axis=1)
            case 'Dependents':
                charts = None if symbols_dependents is None else pd.concat([  v.rename(  columns={c:f"{k},{c}" for c in v}  ) for k, v in symbols_dependents.items()  ], axis=1)
            case 'masks_2nd':
                charts = None if symbols_masks_2nd is None else pd.concat([  v.rename(  columns={c:f"{k},{c}" for c in v}  ) for k, v in symbols_masks_2nd.items()  ], axis=1)
            case 'masks':
                charts = pd.concat([  v for k, v in symbols_masks.items()  ], axis=1)
            case 'OHLCV':
                charts = None if symbols_masks_2nd is None else pd.concat([  v.rename(  columns={c:f"{k},{c}" for c in v}  ) for k, v in symbols_ohlcvs.items()  ], axis=1)
            case 'Benchmark':
                charts = None if symbols_masks_2nd is None else pd.concat([  v.rename(  columns={c:f"{k},{c}" for c in v}  ) for k, v in symbols_benchmarks.items()  ], axis=1)
            case 'Pnl':
                charts = pnls
            case 'Charge':
                charts = charges
            case 'Cpnl':
                charts = cpnls
            case 'TA':
                charts = ects

        if charts is not None and not charts.empty:
            
            match chart_type:
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
                                                if chart_selected in ['Independent', 'primary_mask', 'Start']:
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
                                
        
if __name__ == '__main__':

    app()