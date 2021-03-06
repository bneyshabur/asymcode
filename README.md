Asymmetric Binary Hashing
========
This packages contains the matlab implementation of LIN:LIN and LIN:V algorithms presented
in the following paper:

The Power of Asymmetry in Binary Hashing
Behnam Neyshabur, Payman Yadollahpour, Yury Makarychev, Ruslan Salakhutdinov, Nati Srebro.
Neural Information Processing Systems (NIPS) 26, 2013.

Please note that although there is a default value for parameters, for
some datasets you need to tune them by cross-validation to get the desired result.

Main Files:
**************

LIN:LIN method (linear hash funcitons for both query and database):

  run_LIN_LIN.m: This is just a sample that shows how to run the main function LIN_LIN.m

  LIN_LIN.m: The main function to be called. This function sets some parameters and then calls LIN_LIN_logistic.

  LIN_LIN_logistic.m: The most of the work is done in this function.

  UpdateW.m: This function does the alternating updates of a fixed bit for all objects.

**************

LIN:V method (linear hash funcitons for query and arbitrary hashes for
database):

  run_LIN_V.m: This is just a sample that shows how to run the main function LIN_V.m

  LIN_V.m: The main function to be called. This function sets some parameters and then calls LIN_V_logistic.

  LIN_V_logistic.m: The most of the work is done in this function.

  UpdateWV.m: This function does the alternating updates of a fixed bit for all objects.

**************

We will continue to update the package. So please visit the following website for the latest version of the software:

http://ttic.uchicago.edu/~btavakoli/

If you have any questions, please contact btavakoli@ttic.edu.
