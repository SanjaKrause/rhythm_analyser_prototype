======
maipc
======

Maipc (Madmom applied in [the field of] pulse clarity) is a Python library developed to implement the pulse clarity metrics
presented in [1]_, based on the `madmom <https://github.com/CPJKU/madmom>`_ [2]_ package.

It provides the implementation of all the metrics proposed to be used in python 
programs. It also comes with a script that enables the user to run any 
given metric for an audio file from the command line.   

Installation
============

Prerequisites
-------------

To install the ``maipc`` package, you must have Python 3.6 or newer
installed, as well as the following packages:

- `madmom <https://github.com/CPJKU/madmom>`_
- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_
- `cython <http://www.cython.org>`_
- `mido <https://github.com/olemb/mido>`_
- `KDEpy <https://github.com/tommyod/KDEpy>`_
- `Python fire <https://github.com/google/python-fire>`_

Please refer to the `requirements.txt <requirements.txt>`_ file for the minimum
required versions and make sure that these modules are up to date.


Install from source
-------------------

If you plan to use the package as a developer, clone the Git repository::

    git clone  https://github.com/nPironio/maipc.git

Then you can simply install the package in development mode::

    python setup.py develop --user

Usage
=====

``maipc`` has three main modules:

* ``rnn.py``: provides the ``cell_state_precision``, ``neurons_cross_correlation`` and ``autocorrelation_periodicity`` pulse clarity metric functions.
* ``dbn.py``: provides the ``viterbi_max`` and ``viterbi_entropy`` pulse clarity metric functions.
* ``output.py``: provides the ``peak_average``, ``RNN_entropy`` and ``DBN_entropy`` pulse clarity metric functions.

Each metric is a function with signature ::

	def metric_function(audio_path, downbeat_model=True)

Where ``audio_path`` can be either a string representing an audio file path or a list of paths, and ``downbeat_model`` indicates wether to use the downbeat or beat tracking model implemented in [2]_.

Using the package in a Python script
------------------------------------

You can import each module separately ::

	import maipc.dbn as DBN

	DBN.viterbi_max('my_audio_file.wav')

Or simply import the whole package, which in turn imports the metric functions ::

	import maipc

	maipc.viterbi_max('my_audio_file.wav')


Using the CLI program
---------------------

Included in the ``bin/`` folder comes the ``maipc`` program, which enables to run any metric from a console interface ::

	$ maipc <metric_fn> 'my_audio_file.wav'

Or if you want the result for a list of files ::

	$ maipc viterbi_max '["audio1.wav", "audio2.wav", ... ]'

Licence
=======


References
==========

.. [1] "Pulse clarity metrics developed from a deep learning beat tracking model", Pironio, Nicolás and Miguel, Martín and Slezak, Diego Fernández. Proceedings of the 22nd International Society for Music Information Retrieval Conference, November 2021.
.. [2] "madmom: a new Python Audio and Music Signal Processing Library", Bock, Sebastian and Korzeniowski, Filip and Schluter, Jan and Krebs, Florian and Widmer, Gerhard. Proceedings of the 24th ACM International Conference on Multimedia, October 2016.
