# retail_cleaning_pipeline.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def load_data(path: str) -> pd.DataFrame:
    """
    Load and preprocess invoice data from an Excel file.

    Parameters
    ----------
    path : str
        The file path to the Excel document containing invoice data.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with parsed 'InvoiceDate' as datetime
        and 'InvoiceNo' converted to string.
    """
    df = pd.read_excel(path, parse_dates=["InvoiceDate"])
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    return df

def add_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    return df

def handle_missing_customer_ids(df: pd.DataFrame) -> pd.DataFrame:
    df['CustomerID'] = df.apply(
        lambda row: f"guest_{row['Country']}" if pd.isnull(row['CustomerID']) else row['CustomerID'],
        axis=1
    )
    return df

def remove_invalid_invoices(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df['InvoiceNo'].str.startswith('A')].copy()

def add_subtotal_column(df: pd.DataFrame) -> pd.DataFrame:
    df['Subtotal'] = df['Quantity'] * df['UnitPrice']
    return df

def flag_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate by InvoiceNo
    summary = df.groupby('InvoiceNo').agg({
        'Subtotal': 'sum',
        'StockCode': lambda x: set(x),
        'CustomerID': 'first'  # assumes consistent CustomerID per invoice
    }).reset_index()

    summary = summary.dropna(subset=['InvoiceNo']).copy()
    summary['InvoiceNo'] = summary['InvoiceNo'].astype(str)

    cancelled = summary[summary['InvoiceNo'].str.startswith('C')].copy()
    non_cancelled = summary[~summary['InvoiceNo'].str.startswith('C')].copy()

    match_map = {}  # cancelled → valid
    reverse_map = {}  # valid → cancelled

    for _, c_row in cancelled.iterrows():
        c_inv = c_row['InvoiceNo']
        c_sub = c_row['Subtotal']
        c_codes = c_row['StockCode']
        c_cust = c_row['CustomerID']

        for _, v_row in non_cancelled.iterrows():
            v_inv = v_row['InvoiceNo']
            if (
                c_sub == -v_row['Subtotal'] and
                c_codes == v_row['StockCode'] and
                c_cust == v_row['CustomerID']
            ):
                match_map[c_inv] = v_inv
                reverse_map[v_inv] = c_inv
                break

    def flag(row):
        inv = row['InvoiceNo']
        if inv in match_map:
            return f"matched_with_{match_map[inv]}"
        elif inv in reverse_map:
            return f"matched_with_cancel_{reverse_map[inv]}"
        elif inv.startswith('C'):
            return "cancelled-no-valid"
        return None

    df['CancellationFlag'] = df.apply(flag, axis=1)
    return df


def remove_invalid_quantity_records(df: pd.DataFrame) -> pd.DataFrame:
    return df[~((df['Quantity'] < 0) & (df['UnitPrice'] == 0))].copy()


def remove_price_0(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['UnitPrice'] != 0]

def remove_unspecified_country(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['Country'].notna() & (df['Country'].str.lower() != 'unspecified')]


def clean_retail_data(path: str) -> pd.DataFrame:
    """
    Load and preprocess invoice data from an Excel file.

    Parameters
    ----------
    path : str
        The file path to the Excel document containing invoice data.

    Returns
    -------
    pd.DataFrame
        To adjust
    """
    logging.info("Loading data...")
    df = load_data(path)
    logging.info("Adding date columns...")
    df = add_date_columns(df)
    logging.info("Handling missing CustomerID...")
    df = handle_missing_customer_ids(df)
    logging.info("Removing invalid InvoiceNo entries...")
    df = remove_invalid_invoices(df)
    logging.info("Adding Subtotal column...")
    df = add_subtotal_column(df)
    logging.info("Flagging cancellations...")
    df = flag_cancellations(df)
    logging.info("Removing invalid quantity records...")
    df = remove_invalid_quantity_records(df)
    logging.info("Removing UnitPrice = 0...")
    df = remove_price_0(df)
    logging.info("Removing 'unspecified' Countries...")
    df = remove_unspecified_country(df)
    logging.info("Cleaning complete.")

    return df
