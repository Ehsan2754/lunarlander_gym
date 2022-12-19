===============
lunarlander_gym
===============


.. image:: https://img.shields.io/pypi/v/lunarlander_gym.svg
        :target: https://pypi.python.org/pypi/lunarlander_gym

.. image:: https://img.shields.io/travis/ehsan2754/lunarlander_gym.svg
        :target: https://travis-ci.com/ehsan2754/lunarlander_gym

.. image:: https://readthedocs.org/projects/lunarlander-gym/badge/?version=latest
        :target: https://lunarlander-gym.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/ehsan2754/lunarlander_gym/shield.svg
     :target: https://pyup.io/repos/github/ehsan2754/lunarlander_gym/
     :alt: Updates



This project is implementation of multiple AI agents based on different Reinforcement Learning methods  to OpenAI Gymnasium Lunar-Lander environment which is classic rocket landing trajectory optimization problem.


* Free software: MIT license
* Documentation: https://lunarlander-gym.readthedocs.io.


.. highlight:: shell

============
Installation
============


.. Stable release [Not Available]
.. --------------

.. To install lunarlander_gym, run this command in your terminal:

.. .. code-block:: console

..     $ pip install lunarlander_gym

.. This is the preferred method to install lunarlander_gym, as it will always install the most recent stable release.

.. If you don't have `pip`_ installed, this `Python installation guide`_ can guide
.. you through the process.

.. .. _pip: https://pip.pypa.io
.. .. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for lunarlander_gym can be downloaded from the `Github repo`_.

.. You can either clone the public repository:
* Clone the repository
.. code-block:: console

    $ git clone git://github.com/ehsan2754/lunarlander_gym

.. Or download the `tarball`_:

.. .. code-block:: console

..     $ curl -OJL https://github.com/ehsan2754/lunarlander_gym/tarball/master


* Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ sudo apt update && sudo apt upgrade
    $ sudo apt install make
    $ pip install requirements_dev.txt
    $ make install


Now you can just immidiately use it:

.. code-block:: console
        
    $ lunarlander-gym -h
        usage: lunarlander_gym [-h] -m M

        options:
          -h, --help        show this help message and exit
          -m M, --method M  Specifies the Reinforcement Agent method { 0 -> Random, 1 ->
                            Gradient based optimization, 2 -> Q-Learning Agent 3 -> Actor-
                            critic }
    

Features
--------



Credits
-------


