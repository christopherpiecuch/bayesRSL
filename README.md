# bayesRSL
The computer code used to run the Bayesian model and produce the results in Piecuch et al. (2017), Journal of Geophysical Research-Oceans,  https://doi.org/10.1002/2016JC012506, written in the MATLAB software environment 

bayesRSL

README file last updated by Christopher Piecuch, cpiecuch-at-whoi-dot-edu, Wed Aug 28 2019

Basic description

Citation

This code was generated to produce the results presented in the main text of:

Piecuch, C. G., P. Huybers, and M. P. Tingley, “Comparison of full and empirical Bayes approaches for inferring sea‐level changes from tide‐gauge data”, Journal of Geophysical Research-Oceans, 122(3), 2243-2258, https://doi.org/10.1002/2016JC012506.

Please cite this reference when using this code. 

Contents

Text files

•	Copyright: copyright statement

•	License: license statement

MATLAB .m files

•	autocorrelation.m: computes sample autocorrelation of time series

•	bayes_main_code.m: this is the main driver code of the model.  Simply execute “bayes_main_code” in the MATLAB Command Window from the bayesGIA directory, and this code should run “out of the box”.  See lines 20-34 of this code for adjustable input parameters.  (Values occurring on lines 29-34 presently are the “default” values to reproduce results in Piecuch et al. (2017)).

•	delete_burn_in.m: delete “burn-in” (or “warm-up”) transients from model solution

•	EarthDistances.m: compute distances between latitude and longitude points on a spherical Earth

•	init_vals_pickup.m: initialize output from pickup file

•	initialize_output.m: initialize output

•	prepare_data.m: format tide gauge data and bring into Bayesian model workspace

•	randraw.m: draw random value from any of a number of distributions

•	set_hyperparameters.m: set hyperparameter values (i.e., parameters of the prior distributions)

•	set_initial_values.m: set initial values of model solutions

•	update_all_arrays.m: update model solutions

Subdirectories (each with readme files)

•	rlr_annual: tide gauge records of relative sea level.  Note that these files were downloaed on 2019-08-28, with values extending through year 2018.  This is a more recent dataset than is used in Piecuch et al. (2017), which only included data up through 2015.  These files were downloaded from https://www.psmsl.org/data/obtaining/rlr.annual.data/rlr_annual.zip and are included here for ease and completeness.

Note on usage of code and MATLAB release

This code was written and executed (to produce results in Piecuch et al. 2017) using the MATLAB release version MATLAB_R2015b.  The code is also compatible with the MATLAB_R2016a release.  However, it has come to our attention that there are issues with newer MATLAB releases, such that errors are incurred when the code is executed using MATLAB_R2016b or later.  We are currently investigating this issue and hope to provide updated codes, that are compatible with more recent MATLAB releases, as soon as possible. 
