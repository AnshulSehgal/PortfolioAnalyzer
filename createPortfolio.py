import pandas as pd
import numpy as np
from datetime import datetime, date
from alpha_vantage.timeseries import TimeSeries
from itertools import compress
from nsepy import get_history


# Global variables
filename = 'alerts.csv'
starting_amt = 1000000
interest = 0.06
dt = str(datetime.date(datetime.now()))
api_key = 'HKWHRTVR81SPWWI9'


def update_prices(tickers, stock_history_df, key, start_date, end_date):
    """
    :param tickers:
    :param stock_history_df:
    :param key:
    :param start_date:
    :param end_date:
    :return: An updated stock_history_df with prices till today
    """
    print('---------- Updating Prices for all stocks to current date -------------')
    start_date = date(int(start_date.split("-")[0]), int(start_date.split("-")[1]), int(start_date.split("-")[2]))
    end_date = date(int(end_date.split("-")[0]), int(end_date.split("-")[1]), int(end_date.split("-")[2]))

    #ts = TimeSeries(key, output_format='pandas', retries=5)
    for ticker in tickers:
        tk = ticker.split(".")[0]
        print('---------- Fetching prices for {} -----------'.format(ticker))
        tkr = get_history(symbol=tk, start=start_date, end=end_date)
        #tkr, meta = ts.get_daily(symbol=ticker)
        tkr = tkr.reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        tkr.columns = ['DATE', 'open', 'high', 'low', 'CLOSING_RATES', 'volume']
        tkr['TICKER'] = ticker
        tkr['DATE'] = pd.to_datetime(tkr['DATE']).dt.strftime('%Y-%m-%d')
        stock_history_df = stock_history_df.append(tkr)
        stock_history_df.drop_duplicates(inplace=True)
        #time.sleep(12)
    return stock_history_df


def get_month_ends(start_date, history):
    """
    :param start_date: a date object
    :param history: a list of historical dates of prices from NSE
    :return: a list of dates for all month ends between start date and end date
    """
    history_mon = [val[0:7] for val in history]

    dates = pd.DataFrame({'DATE': history, 'DATE_MON': history_mon})

    dates = dates.groupby('DATE_MON').DATE.max().to_list()
    fil = [dt >= start_date for dt in dates]

    return list(compress(dates, fil))


def insert_month_ends(df, m):
    """
    :param df: a dataframe with alerts and prices
    :param m: a list of month ends from starting month of alerts to last month of alerts
    :return: a dataframe with 1 additional row for each month end
    """
    df['month_end'] = 0
    to_append = pd.DataFrame()
    for item in m:
        to_append = to_append.append(pd.DataFrame({'TICKER': [None], 'DATE': [item],
                                                   'TYPE': [None], 'QUANTITY': [None],
                                                   'open': [None], 'close': [None], 'high': [None],
                                                   'low': [None], 'volume': [None],
                                                   'avg_price': [None], 'month_end': [1]}))
    df = df.append(to_append, ignore_index=True)
    df.sort_values(by=['DATE', 'month_end'], inplace=True)
    return df


def generate_transactions(alerts_df, stock_history_df, key, initial_amount, int_rate):
    """
    :param alerts_df: a pandas dataframe of alerts
    :param stock_history_df: dataframe of historical stock prices
    :param key: API key for alpha vantage
    :param initial_amount: Initial amount to start with
    :param int_rate: Annual interest rate for amount in bank
    :return: a dataframe with all transactions and month end value of investments
             and a dataframe with historical prices udpated
    """
    # Load the stock prices
    #ts = TimeSeries(key, output_format='pandas')
    stock_history_df = stock_history_df.sort_values(by=['DATE'], ascending=False)
    df_prices = pd.DataFrame()
    tickers = alerts_df.TICKER.unique()
    print('---------- Adding stock prices to alerts -------------')
    for ticker in tickers:
        print('---- checking history data for {} ------'.format(ticker))
        dt = alerts_df['DATE'][alerts_df['TICKER'] == ticker].unique()
        if (stock_history_df[(stock_history_df['TICKER'] == ticker) &
                             (stock_history_df['DATE'].isin(dt))].shape[0] == len(dt)):
            prices = stock_history_df[(stock_history_df['TICKER'] == ticker) & (stock_history_df['DATE'].isin(dt))]
            prices = alerts_df[alerts_df['TICKER'] == ticker].merge(prices, on=['DATE', 'TICKER'], how='left')
            df_prices = df_prices.append(prices)
        else:
            quit("Price not available for ticker {}".format(ticker))

        ''' This is not required anymore
        # check if data already exists in the stock history
        if stock_history_df.shape[0] > 0:
            if (stock_history_df[(stock_history_df['TICKER'] == ticker) &
                                 (stock_history_df['DATE'].isin(dt))].shape[0] == len(dt)):
                print('-------- Data available in stock history for {} loading from history ----------'.format(ticker))
                prices = stock_history_df[
                    (stock_history_df['TICKER'] == ticker) & (stock_history_df['DATE'].isin(dt))]
                prices = alerts_df[alerts_df['TICKER'] == ticker].merge(prices, on=['DATE', 'TICKER'], how='left')
                df_prices = df_prices.append(prices)
            else:
                print('---------- Fetching prices for {} -----------'.format(ticker))
                tkr_dta, meta_data = ts.get_daily(symbol=ticker)
                tkr_dta = tkr_dta.reset_index()
                tkr_dta.columns = ['DATE', 'open', 'high', 'low', 'CLOSING_RATES', 'volume']
                tkr_dta['TICKER'] = ticker
                prices = tkr_dta[tkr_dta['DATE'].isin(dt)]
                stock_history_df = stock_history_df.append(tkr_dta)
                stock_history_df.drop_duplicates(inplace=True)
                df_prices = df_prices.append(
                    alerts_df[alerts_df['TICKER'] == ticker].merge(prices,
                                                                   on=['DATE', 'TICKER'],
                                                                   how='left'))
                time.sleep(12)
        else:
            print('---------- Fetching prices for {} -----------'.format(ticker))
            tkr_dta, meta_data = ts.get_daily(symbol=ticker)
            tkr_dta = tkr_dta.reset_index()
            tkr_dta.columns = ['DATE', 'open', 'high', 'low', 'CLOSING_RATES', 'volume']
            tkr_dta['TICKER'] = ticker
            prices = tkr_dta[tkr_dta['DATE'].isin(dt)]
            stock_history_df = stock_history_df.append(tkr_dta)
            stock_history_df.drop_duplicates(inplace=True)
            df_prices = df_prices.append(
                alerts_df[alerts_df['TICKER'] == ticker].merge(prices, on=['DATE', 'TICKER'], how='left'))
            time.sleep(12)
    '''
    # add price column which is average of min and max
    df_prices['avg_price'] = df_prices.apply(lambda row: np.mean([row['high'], row['low']]), axis=1)

    month_ends = get_month_ends(start_date=min(df_prices['DATE']),
                                history=stock_history_df['DATE'].unique().tolist())

    df_prices = insert_month_ends(df_prices, month_ends)

    '''Calculating the amount invested and amount in bank 
    Step 1: Start with the amount_in_bank 
    Step 2: For every alert starting from the oldest date calculate the invested amount as 
            avg_price *  quantity Step 3: If type is BUY 
            and invested amount calculated is less than amount_in_bank then 
            amount_in_bank = amount_in_bank - invested_amount 
            If type is SELL then 
            amount_in_bank = amount_in_bank + invested_amount & 
            invested_amount = invested_amount - amount invested in the current stock '''
    amount_in_bank = [initial_amount]
    cumulative_invested_amount = []
    invested_amount = []
    closing_value = []
    open_position_eod = []
    stocks = dict()
    print('---------- Calculating Invested amount and amount in bank after each transaction -------------')
    for _, row in df_prices.iterrows():

        act = row['TYPE']
        tkr_dta = row['TICKER']
        qty = row['QUANTITY']
        # price = row['avg_price']
        price = row['RATE']
        m_e = row['month_end']
        print('{} and {}'.format(tkr_dta, row['DATE']))
        if m_e == 0:
            if act == 'SELL':
                if tkr_dta not in stocks.keys():
                    avl_qty = 0
                else:
                    if stocks[tkr_dta]['qty'] >= qty:
                        avl_qty = qty
                    else:
                        avl_qty = stocks[tkr_dta]['qty']
                    cash_out = (avl_qty / stocks[tkr_dta]['qty']) * stocks[tkr_dta]['amt']
                    stocks[tkr_dta]['amt'] = (1 - avl_qty / stocks[tkr_dta]['qty']) * stocks[tkr_dta]['amt']
                    stocks[tkr_dta]['qty'] = stocks[tkr_dta]['qty'] - avl_qty
                investment = price * avl_qty
                if len(cumulative_invested_amount) > 0:
                    cumulative_invested_amount.append(cumulative_invested_amount[-1] - cash_out)
                    amount_in_bank.append(amount_in_bank[-1] + investment)
                    invested_amount.append(investment)
                    closing_value.append(row['CLOSING_RATES'] * avl_qty)
                else:
                    cumulative_invested_amount.append(cash_out)
                    amount_in_bank = [amount_in_bank[0] + investment]
                    invested_amount.append(investment)
                    closing_value.append(row['CLOSING_RATES'] * avl_qty)
            else:
                if (price * qty) > amount_in_bank[-1]:
                    avl_qty = 0
                else:
                    avl_qty = qty
                investment = price * avl_qty
                if tkr_dta not in stocks.keys():
                    stocks[tkr_dta] = {'qty': avl_qty, 'amt': investment}
                else:
                    stocks[tkr_dta]['qty'] = stocks[tkr_dta]['qty'] + avl_qty
                    stocks[tkr_dta]['amt'] = stocks[tkr_dta]['amt'] + investment
                if len(cumulative_invested_amount) > 0:
                    cumulative_invested_amount.append(cumulative_invested_amount[-1] + investment)
                    amount_in_bank.append(amount_in_bank[-1] - investment)
                    invested_amount.append(investment)
                    closing_value.append(row['CLOSING_RATES'] * avl_qty)
                else:
                    cumulative_invested_amount.append(investment)
                    amount_in_bank = [amount_in_bank[-1] - investment]
                    invested_amount.append(investment)
                    closing_value.append(row['CLOSING_RATES'] * avl_qty)
        else:
            cumulative_invested_amount.append(cumulative_invested_amount[-1])
            amount_in_bank.append(amount_in_bank[-1])
            row_count = df_prices.loc[df_prices['DATE'].str[:7] == row['DATE'][:7]].shape[0]
            interest = min(amount_in_bank[-row_count:]) * (int_rate / 12)
            amount_in_bank[-1] = amount_in_bank[-1] + interest
            invested_amount.append(0)
            closing_value.append(0)
        open_pos = {stk: pos for (stk, pos) in stocks.items() if pos['qty'] > 0}
        eod = 0
        for stk, pos in open_pos.items():
            eod += stock_history_df[(stock_history_df['TICKER'] == stk) &
                                    (stock_history_df['DATE'] <= row['DATE'])].head(1)['CLOSING_RATES'].values[0] * pos[
                       'qty']
        open_position_eod.append(eod)

    df_prices['AMOUNT_IN_BANK'] = amount_in_bank
    df_prices['cumulative_amount_invested'] = cumulative_invested_amount
    df_prices['CUMULATIVE_VALUE_OF_STOCKS_AT_CLOSING'] = open_position_eod
    df_prices['invested_amount'] = invested_amount
    df_prices['closing_value_of_stock'] = closing_value
    df_prices['VALUE_OF_PORTFOLIO_AT_CLOSING'] = df_prices['AMOUNT_IN_BANK'] + df_prices[
        'CUMULATIVE_VALUE_OF_STOCKS_AT_CLOSING']

    return df_prices, stock_history_df


def load_data(alerts_filename):
    # Load the alerts and historical data
    try:
        print('---------- Loading alerts -----------')
        alerts = pd.read_csv('input/{}'.format(alerts_filename))
        alerts['DATE'] = pd.to_datetime(alerts['DATE']).dt.strftime('%Y-%m-%d')
    except FileNotFoundError:
        print("-------- The alert file dosen't exist. Please store the file in project folder and re-run -----------")
        exit()

    try:
        print('---------- Loading price history -----------')
        stock_history = pd.read_csv('stock_history.csv')
        stock_history['DATE'] = pd.to_datetime(stock_history['DATE']).dt.strftime('%Y-%m-%d')
    except FileNotFoundError:
        print("------- Stock history not available ... It will be created now ----------")
        stock_history = pd.DataFrame()

    return alerts, stock_history


def get_nifty(key):
    ts = TimeSeries(key, output_format='pandas')
    nfty, meta = ts.get_monthly(symbol='^NSEI')
    nfty = nfty.reset_index()
    nfty.columns = ['DATE', 'NIFTY_50_VALUE_AT_OPENING', 'high', 'low', 'NIFTY_50_VALUE_AT_CLOSING', 'volume']
    nfty['NIFTY_RETURNS'] = (nfty['NIFTY_50_VALUE_AT_CLOSING'] - nfty['NIFTY_50_VALUE_AT_OPENING']) \
                            / nfty['NIFTY_50_VALUE_AT_OPENING']
    # nfty['TICKER'] = 'NIFTY 50'
    return nfty


def analyze_portfolio(trans, idx, init_amt, bank_int):
    # Formatting as per requirements
    trans['MONTH'] = pd.to_datetime(trans['DATE']).dt.to_period('m')
    col_order = ['MONTH', 'TICKER', 'DATE', 'TYPE', 'RATE', 'QUANTITY',
                 'invested_amount', 'cumulative_amount_invested', 'AMOUNT_IN_BANK',
                 'CLOSING_RATES', 'closing_value_of_stock', 'VALUE_OF_PORTFOLIO_AT_CLOSING',
                 'CUMULATIVE_VALUE_OF_STOCKS_AT_CLOSING', 'avg_price', 'high', 'low', 'month_end', 'open', 'volume']

    trans = trans[col_order]

    # calculate min amount in bank every month
    min_amt = trans[trans['month_end'] != 1].groupby('MONTH')['AMOUNT_IN_BANK'].min().reset_index()
    min_amt.columns = ['MONTH', 'MINIMUM_BALANCE_IN_BANK']
    min_amt['INTEREST_ON_MINIMUM_BALANCE_AT_{:.2f}_%'.format((bank_int / 12) * 100)] = \
        min_amt['MINIMUM_BALANCE_IN_BANK'] * bank_int / 12

    m_cols = ['MONTH', 'DATE', 'CUMULATIVE_VALUE_OF_STOCKS_AT_CLOSING',
              'VALUE_OF_PORTFOLIO_AT_CLOSING', 'AMOUNT_IN_BANK']
    portfolio_stats = trans.loc[trans['month_end'] == 1, m_cols]
    portfolio_stats['VALUE_OF_PORTFOLIO_AT_OPENING'] = portfolio_stats['VALUE_OF_PORTFOLIO_AT_CLOSING'].shift().fillna(
        init_amt)
    portfolio_stats['10X_RETURNS'] = (portfolio_stats['VALUE_OF_PORTFOLIO_AT_CLOSING']
                                      - portfolio_stats['VALUE_OF_PORTFOLIO_AT_OPENING']) / portfolio_stats[
                                         'VALUE_OF_PORTFOLIO_AT_OPENING']

    portfolio_stats = portfolio_stats.merge(min_amt, on=['MONTH'])
    portfolio_stats = portfolio_stats.merge(idx[['DATE', 'NIFTY_50_VALUE_AT_OPENING',
                                                 'NIFTY_50_VALUE_AT_CLOSING', 'NIFTY_RETURNS']], on='DATE')

    m_col_order = ['MONTH', 'CUMULATIVE_VALUE_OF_STOCKS_AT_CLOSING', 'AMOUNT_IN_BANK',
                   'MINIMUM_BALANCE_IN_BANK', 'INTEREST_ON_MINIMUM_BALANCE_AT_{:.2f}_%'.format((bank_int / 12) * 100),
                   'VALUE_OF_PORTFOLIO_AT_OPENING', 'VALUE_OF_PORTFOLIO_AT_CLOSING',
                   '10X_RETURNS', 'NIFTY_50_VALUE_AT_OPENING', 'NIFTY_50_VALUE_AT_CLOSING', 'NIFTY_RETURNS']

    portfolio_stats = portfolio_stats[m_col_order]

    return trans, portfolio_stats


alerts, stock_history = load_data(alerts_filename=filename)

stock_history = update_prices(tickers=alerts['TICKER'].unique().tolist(),
                              stock_history_df=stock_history,
                              key=api_key,
                              start_date=alerts.DATE.min(),
                              end_date=dt)

stock_history.to_csv('archive/stock_history_{}.csv'.format(dt), index=False)


df_trans, stock_history = generate_transactions(alerts_df=alerts, stock_history_df=stock_history,
                                                key=api_key,
                                                initial_amount=starting_amt,
                                                int_rate=interest)

# Get index prices
nifty = get_nifty(key=api_key)
nifty.to_csv('output/NIFTY_50_{}.csv'.format(dt), index=False)


portfolio, roi = analyze_portfolio(df_trans, nifty, starting_amt, interest)

portfolio.to_csv('output/portfolio_{}.csv'.format(dt), index=False)
roi.to_csv('output/roi_{}.csv'.format(dt), index=False)
