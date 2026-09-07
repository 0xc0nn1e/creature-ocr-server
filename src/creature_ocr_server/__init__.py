"""The OCR API for the handwritten wildlife survey sheets.

The desktop application crops the personal-information band off a scan and
sends the page here; this package decides what is written on it. Everything
that needs a cloud credential, a model id or a prompt lives on this side, so an
operator's machine has nothing to configure. (4.2, 6.2, 6.5)
"""

__version__ = "0.1.0"
