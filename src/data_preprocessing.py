import re
import math
import logging
from typing import Tuple, Optional, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Optional libraries: quantulum3 (quantity parser) and pint (unit conversions)
try:
    from quantulum3 import parser as quantulum_parser
    HAS_QUANTULUM = True
    logger.info("quantulum3 available")
except Exception:
    HAS_QUANTULUM = False
    logger.info("quantulum3 NOT available")

try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    HAS_PINT = True
    logger.info("pint available")
except Exception:
    HAS_PINT = False
    logger.info("pint NOT available")

# Basic fallback multipliers to convert common units into base units:
# mass -> grams (g), volume -> milliliters (ml)
UNIT_MULTIPLIERS = {
    'mg': ('mass', 0.001), 'milligram': ('mass', 0.001),
    'g': ('mass', 1.0), 'gram': ('mass', 1.0), 'grams': ('mass', 1.0),
    'kg': ('mass', 1000.0), 'kilogram': ('mass', 1000.0),
    'lb': ('mass', 453.59237), 'lbs': ('mass', 453.59237), 'pound': ('mass', 453.59237),
    'oz': ('mass', 28.349523125), 'ounce': ('mass', 28.349523125), 'ounces': ('mass', 28.349523125),
    'ml': ('volume', 1.0), 'milliliter': ('volume', 1.0), 'millilitre': ('volume', 1.0),
    'l': ('volume', 1000.0), 'liter': ('volume', 1000.0), 'litre': ('volume', 1000.0),
    'fl oz': ('volume', 29.5735295625), 'floz': ('volume', 29.5735295625), 'fluid ounce': ('volume', 29.5735295625),
}


# ------------------ User-provided extractor (kept as-is) ------------------
def get_product_info(text):
    """
    Simple helper that extracts trailing `Value:` and `Unit:` fields from text.

    Returns (value_str, unit_str, remaining_text_before_value).
    If fields are absent returns (None, None, original_text).
    """
    # Find last occurrences
    value_pos = text.rfind("Value:")
    unit_pos = text.rfind("Unit:")

    value = unit = None
    remaining_text = text  # default (if Value/Unit not found)

    if value_pos != -1:
        # Extract Value
        value_end = unit_pos if unit_pos > value_pos else len(text)
        value = text[value_pos + len("Value:"):value_end].strip()

    if unit_pos != -1:
        # Extract Unit
        unit = text[unit_pos + len("Unit:"):].strip()

    # Get remaining text before Value:
    if value_pos != -1:
        remaining_text = text[:value_pos].rstrip()

    # Return extracted pieces (no further parsing here)
    return value,unit,remaining_text


# ------------------ Regex helpers & patterns ------------------
# Matches numbers with optional `x` multipliers and common units
QUANTITY_RE = re.compile(r"(?P<num>\d+(?:[\.,]\d+)?)(?:\s*(?:x|×|by)\s*(?P<num2>\d+(?:[\.,]\d+)?))?\s*(?P<unit>mg|g|kg|lbs?|lb|oz|ounces?|ounce|ml|l|litre|liter|fl ?oz|floz|pack|ct|count|pcs|piece|pieces)?\b",
                         flags=re.I)
# Matches explicit "pack of 6"-style patterns
PACK_RE = re.compile(r'(pack of|pack|packs|ct\.|ct\b|count of|count)\s*(?P<p>\d+)', flags=re.I)
# Value/Unit markers (used by user extractor)
VALUE_RE = re.compile(r'Value:\s*([^\n\r]*)', flags=re.I)
UNIT_RE = re.compile(r'Unit:\s*([^\n\r]*)', flags=re.I)


# ------------------ Parsing backends (quantulum / pint) ------------------
def _try_quantulum_parse(text: str):
    """
    Use quantulum3 to extract quantities if available. Returns a list of
    parsed {'value', 'unit'} dicts. Empty list if unavailable or parsing fails.
    """
    if not HAS_QUANTULUM:
        return []
    try:
        qs = quantulum_parser.parse(text)
        parsed = [{'value': q.value, 'unit': str(q.unit.name)} for q in qs]
        return parsed
    except Exception:
        return []


def _pint_convert(value: float, unit_str: str) -> Optional[Dict[str, Any]]:
    """
    Convert (value, unit_str) into a normalized dict using pint when available.

    Returns {'amount': <float>, 'amount_unit': 'g'|'ml'|'count'|'other', 'amount_type': 'mass'|'volume'|'other'}
    or None on failure.
    """
    if not HAS_PINT:
        return None
    try:
        q = value * ureg(unit_str)
        if q.check('[mass]'):
            grams = q.to('gram').magnitude
            return {'amount': float(grams), 'amount_unit': 'g', 'amount_type': 'mass'}
        if q.check('[volume]'):
            ml = q.to('milliliter').magnitude
            return {'amount': float(ml), 'amount_unit': 'ml', 'amount_type': 'volume'}
        # fallback: return raw magnitude + unit
        return {'amount': float(q.magnitude), 'amount_unit': unit_str, 'amount_type': 'other'}
    except Exception:
        return None


def _fallback_unit_norm(value: float, unit_str: Optional[str]):
    """
    Heuristic normalizer for common unit strings when pint is unavailable or fails.

    Returns a dict similar to _pint_convert or None if unknown.
    """
    if not unit_str:
        return None
    s = unit_str.lower().strip()
    s = s.replace('.', '')
    s = s.replace('\u00a0', ' ')
    s = s.replace('fluid ounce', 'fl oz')
    s = s.replace('ounces', 'oz')
    s = s.replace('ounce', 'oz')
    s = s.replace('milliliters', 'ml')
    s = s.replace('liters', 'l')
    s = s.replace('litres', 'l')
    s = s.replace('pack(s)', 'pack')

    # normalize spacing
    s = re.sub(r'\s+', ' ', s)

    # common key mapping
    if s in ('floz', 'fl oz'):
        s = 'fl oz'

    if s in UNIT_MULTIPLIERS:
        typ, mult = UNIT_MULTIPLIERS[s]
        base_value = value * mult
        return {'amount': float(base_value), 'amount_unit': 'g' if typ=='mass' else 'ml', 'amount_type': typ}

    if s in ('count', 'ct', 'pcs', 'piece', 'pieces', 'pack'):
        return {'amount': float(value), 'amount_unit': 'count', 'amount_type': 'count'}

    # last resort: strip non-alpha and try again
    s2 = re.sub(r'[^a-z ]', '', s)
    if s2 in UNIT_MULTIPLIERS:
        typ, mult = UNIT_MULTIPLIERS[s2]
        base_value = value * mult
        return {'amount': float(base_value), 'amount_unit': 'g' if typ=='mass' else 'ml', 'amount_type': typ}

    return None


# ------------------ High-level parsing using user's extractor + fallbacks ------------------
def parse_quantity_from_text_using_user_parser(text: str) -> Dict[str, Any]:
    """
    Attempt to extract an "amount" and its unit/type from free-form product text.

    Strategy (priority order):
      1. Use user's `Value:` / `Unit:` extractor (if present).
      2. Try quantulum3 parsing.
      3. Regex patterns (numbers, optional multipliers, units).
      4. Pack-count heuristics ("pack of 6").

    Returns a dict with keys: amount, amount_unit, amount_type, raw_unit, pack_count, original_value.
    Unset fields remain None.
    """
    out = {'amount': None, 'amount_unit': None, 'amount_type': None, 'raw_unit': None, 'pack_count': None, 'original_value': None}
    if not isinstance(text, str) or not text:
        return out

    # 1. Use user's logic first
    value_str, unit_str, remaining = get_product_info(text)
    if value_str:
        out['original_value'] = value_str
        # handle patterns like '2 x 250' or single numeric
        m = re.search(r'(?P<n1>\d+(?:[\.,]\d+)?)(?:\s*(?:x|×)\s*(?P<n2>\d+(?:[\.,]\d+)?))', value_str)
        if m:
            n1 = float(m.group('n1').replace(',', '.'))
            n2 = float(m.group('n2').replace(',', '.'))
            numeric_val = n1 * n2
        else:
            try:
                numeric_val = float(value_str.replace(',', '.'))
            except Exception:
                numeric_val = None

        if numeric_val is not None:
            if unit_str:
                out['raw_unit'] = unit_str
                conv = None
                if HAS_PINT:
                    conv = _pint_convert(numeric_val, unit_str)
                if conv is None:
                    conv = _fallback_unit_norm(numeric_val, unit_str)
                if conv:
                    out.update(conv)
                    return out
            else:
                # no unit provided but numeric value exists -> treat as count
                out['amount'] = numeric_val
                out['amount_unit'] = 'count'
                out['amount_type'] = 'count'
                return out

    # 2. quantulum
    if HAS_QUANTULUM:
        qs = _try_quantulum_parse(text)
        if qs:
            first = qs[0]
            out['original_value'] = first['value']
            out['raw_unit'] = first['unit']
            if HAS_PINT:
                conv = _pint_convert(first['value'], first['unit'])
                if conv:
                    out.update(conv)
                    return out
            conv = _fallback_unit_norm(first['value'], first['unit'])
            if conv:
                out.update(conv)
                return out

    # 3. regex pattern search
    for m in QUANTITY_RE.finditer(text):
        n1 = float(m.group('num').replace(',', '.'))
        n2 = m.group('num2')
        unit = m.group('unit')
        if n2:
            n2 = float(n2.replace(',', '.'))
            nval = n1 * n2
        else:
            nval = n1
        if unit:
            conv = None
            if HAS_PINT:
                conv = _pint_convert(nval, unit)
            if conv is None:
                conv = _fallback_unit_norm(nval, unit)
            if conv:
                out.update(conv)
                out['raw_unit'] = unit
                return out
        else:
            # no unit -> interpret as count
            out['amount'] = nval
            out['amount_unit'] = 'count'
            out['amount_type'] = 'count'
            out['original_value'] = str(nval)
            return out

    # 4. pack patterns like "pack of 6"
    m = PACK_RE.search(text)
    if m:
        try:
            p = int(m.group('p'))
            out['pack_count'] = p
            out['amount'] = float(p)
            out['amount_unit'] = 'count'
            out['amount_type'] = 'count'
            return out
        except Exception:
            pass

    return out


# ------------------ Dataframe preprocessing wrapper ------------------
def preprocess_df_using_user_parser(df: pd.DataFrame, text_col: str = 'catalog_content') -> pd.DataFrame:
    """
    Apply `parse_quantity_from_text_using_user_parser` row-wise and add canonical columns:
      - amount, amount_unit, amount_type, pack_count, raw_unit
      - amount_per_unit, amount_per_unit_log1p
      - flags: has_quantity, is_weight, is_volume, is_count

    The function preserves the input dataframe and returns a new copy with added columns.
    """
    df = df.copy()
    df['value_extracted'] = None
    df['unit_extracted'] = None
    df['remaining_text'] = None
    df['amount'] = np.nan
    df['amount_unit'] = None
    df['amount_type'] = None
    df['pack_count'] = np.nan
    df['raw_unit'] = None

    for idx, row in df.iterrows():
        text = row.get(text_col, '') if text_col in row else ''
        # apply user's extractor
        val, unit, remaining = get_product_info(text)
        df.at[idx, 'value_extracted'] = val
        df.at[idx, 'unit_extracted'] = unit
        df.at[idx, 'remaining_text'] = remaining

        parsed = parse_quantity_from_text_using_user_parser(text)
        if parsed.get('amount') is not None:
            df.at[idx, 'amount'] = parsed.get('amount')
        if parsed.get('amount_unit'):
            df.at[idx, 'amount_unit'] = parsed.get('amount_unit')
        if parsed.get('amount_type'):
            df.at[idx, 'amount_type'] = parsed.get('amount_type')
        if parsed.get('pack_count'):
            df.at[idx, 'pack_count'] = parsed.get('pack_count')
        if parsed.get('raw_unit'):
            df.at[idx, 'raw_unit'] = parsed.get('raw_unit')

    # compute amount_per_unit (safe handling for missing/zero values)
    df['pack_count'] = df['pack_count'].fillna(1)
    df['pack_count'] = df['pack_count'].replace(0, 1)
    df['amount_per_unit'] = df['amount'] / df['pack_count']
    df['amount_per_unit'] = df['amount_per_unit'].replace([np.inf, -np.inf], np.nan)
    df['amount_per_unit_log1p'] = np.log1p(df['amount_per_unit'].fillna(0.0))

    # helpful boolean flags derived from parsing
    df['has_quantity'] = ~df['amount'].isna()
    df['is_weight'] = df['amount_type'] == 'mass'
    df['is_volume'] = df['amount_type'] == 'volume'
    df['is_count'] = df['amount_type'] == 'count'

    return df


# ------------------ CLI entrypoint for quick preprocessing ------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/DATA/ml_c/dataset/test.csv')
    parser.add_argument('--text_col', type=str, default='catalog_content')
    parser.add_argument('--output', type=str, default='/DATA/ml_c/data/test_preprocessed.csv')
    args = parser.parse_args()

    logger.info('Loading %s', args.input)
    df = pd.read_csv(args.input)
    df_out = preprocess_df_using_user_parser(df, text_col=args.text_col)
    logger.info('Saving to %s', args.output)
    df_out.to_csv(args.output, index=False)

    total = len(df_out)
    has_q = int(df_out['has_quantity'].sum())
    logger.info('Total rows: %d  Rows with quantity parsed: %d (%.6f)', total, has_q, has_q/total if total else 0.0)

    sample = df_out.loc[df_out['has_quantity']].head(10)
    cols = ['sample_id'] if 'sample_id' in df_out.columns else []
    cols += ['value_extracted', 'unit_extracted', 'amount', 'amount_unit', 'pack_count', 'amount_per_unit']
    print(sample[cols].to_string(index=False))
