.. highlight:: shell

============
Installation
============


Stable release [Not Available]
--------------

.. To install lunarlander_gym, run this command in your terminal:

.. .. code-block:: console

..     $ pip install lunarlander_gym

.. This is the preferred method to install lunarlander_gym, as it will always install the most recent stable release.

.. If you don't have `pip`_ installed, this `Python installation guide`_ can guide
.. you through the process.

.. .. _pip: https://pip.pypa.io
.. .. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/






From sources
~~~~~~~~~~~~

The sources for lunarlander_gym can be downloaded from the
``Github repo``\ \_.

-  Clone the repository

   ::

          $ git clone git://github.com/ehsan2754/lunarlander_gym

Or download the `tarball`_:

   ::

    $ curl -OJL https://github.com/ehsan2754/lunarlander_gym/tarball/master


-  Once you have a copy of the source, you can install it with:

   ::

          $ sudo apt update && sudo apt upgrade
          $ sudo apt install make
          $ pip install -r requirements_dev.txt
          $ sudo make install

-  Now you can just immidiately use it:

   ::

          $ lunarlander-gym -h
              usage: lunarlander_gym [-h] -m M

              options:
              -h, --help        show this help message and exit
              -m M, --method M  Specifies the Reinforcement Agent method { 0 -> Random, 1 ->
                                  Gradient based optimization, 2 -> Q-Learning Agent 3 -> Actor-
                                  critic }

.. _Github repo: https://github.com/ehsan2754/lunarlander_gym
.. _tarball: https://github.com/ehsan2754/lunarlander_gym/tarball/master