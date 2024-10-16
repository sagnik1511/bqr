"""Decoder Module"""

from typing import Union, Dict


def decode_values(row: Dict[str, str]) -> Dict[str, Union[int, float, str]]:
    """Decoded values into proper format

    Args:
        row (Dict[str, str]): Raw Information

    Returns:
        Dict[str, Union[int, float, str]]: Decoded information
    """
    final_row = {}
    for k, v in row.items():
        try:
            v = int(v)
            final_row[k] = v
        except:
            try:
                v = float(v)
                final_row[k] = v
            except:
                final_row[k] = v

    return final_row
