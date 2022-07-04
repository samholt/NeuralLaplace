# Configuration file for the Sphinx documentation builder.

# -- Project information
import torchlaplace
project = 'TorchLaplace'
copyright = '2021, Sam Holt'
author = 'Sam Holt'

version = torchlaplace.__version__
release = version

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx_panels',
    'nbsphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = [
    "torchlaplace.css",
]


# -- Options for EPUB output
epub_show_urls = 'footnote'
