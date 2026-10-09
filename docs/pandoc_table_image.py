#!/usr/bin/env python3
from pandocfilters import walk, toJSONFilter, Image, RawInline
import sys

def raise_image(key, val, fmt, meta):
    if key == 'Image' and fmt == 'latex':
        return [
            RawInline('tex', r'\raisebox{-.5\height}{'),
            Image(*val),
            RawInline('tex', '}')
        ]

def center_table_images(key, val, fmt, meta):
    if key == 'Table' and fmt == 'latex':
        return {'t': 'Table', 'c': walk(val, raise_image, fmt, meta)}


if __name__ == "__main__":
    toJSONFilter(center_table_images)
