Installation
============

Requirements
------------

TabEval requires Python 3.10 or later. The framework has been tested on:

* Python 3.10, 3.11, 3.12
* Linux, macOS, and Windows

Installation Methods
--------------------

Install from PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install tabeval

Install from Source
~~~~~~~~~~~~~~~~~~~

For the latest development version or if you want to contribute:

.. code-block:: bash

   git clone https://github.com/SilenceX12138/TabEval.git
   cd TabEval
   pip install -e .

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

The following packages are automatically installed with TabEval:

* ``pandas`` - Data manipulation and analysis
* ``numpy`` - Numerical computing
* ``scikit-learn`` - Machine learning library
* ``torch`` - Deep learning framework
* ``xgboost`` - Gradient boosting framework
* ``scipy`` - Scientific computing
* ``tqdm`` - Progress bars
* ``loguru`` - Logging
* ``pydantic`` - Data validation

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Some plugins require additional dependencies:

All Dependencies
~~~~~~~~~~~~~~~~

To install all optional dependencies:

.. code-block:: bash

   pip install tabeval[all]

