# pyBAKS: python implementation of Bayesian Adaptive Kernel Smoother (BAKS) for estimation of neuronal firing rate

Daniel Svedberg, Brandeis University (2023)

Based on: github.com/nurahmadi/BAKS/ by Nur Ahmadi, 2017

Ahmadi, Nur, Timothy G. Constandinou, and Christos-Savvas Bouganis. "Estimation of neuronal firing rate using Bayesian Adaptive Kernel Smoother (BAKS)." Plos one 13.11 (2018): e0206794

## Introduction
BAKS is a non-parametric method for estimating the firing rate of a neuron from a spike train. It is based on a Bayesian framework and uses an adaptive-width kernel to smooth the spike train. The method is described in detail in the paper above.

pyBAKS is a python implementation of the original BAKS algorithm written by Nur Ahmadi, with added convenience functions for automatically handling various data structures, optimizing the alpha parameter, and parallel-processing of multiple spike arrays simultaneously. 

## Installation & Loading
As of October 26 2023, please install pyBAKS by cloning this repository. The package will be available on PyPI soon. 
You can then load the package by navigating to the directory containing the repository, and then import pyBAKS from within a python console or script:
```python
import pyBAKS
```
## Usage

### Base functions
The base/main function of pyBAKS is `pyBAKS.baks()`. It takes a vector of spiketimes, a corresponding time-index vector, and the alpha parameter as input, and returns a firing rate estimate, and the kernel bandwidth for each index: 
```python  
import pyBAKS
import numpy as np

spike_times = [0.1, 0.3, 0.5]
time = np.arange(0, 1, 0.01)
alpha = 4
BAKS_firing_rate, kernel_bandwidth = pyBAKS.baks(spike_times, time, alpha)
```

Because alpha is a parameter that must be tuned, we have included a function `pyBAKS.optimize_alpha_MLE()` that takes a spike train and time vector as input, and returns the alpha value estimated to maximize the log-likelihood (Maximum Likelihood Estimation/MLE) of the input spike train given the firing rate estimate.

```python
import pyBAKS

Spikes, Rates, Time = pyBAKS.BAKS_test.BAKS_test_data() # generate a 5s/5000-index poisson spike train with 4 variations in firing rate
results_df, best_alpha, best_BAKSrates = pyBAKS.optimize_alpha_MLE(Spikes, Time) # optimize alpha for the spike train

#results_df is a pandas dataframe containing the log-likelihood for each alpha value tested
#best_alpha is the alpha value that maximizes the log-likelihood
#best_BAKSrates is the firing rate estimate for the spike train using the best_alpha value
```

### autoBAKS: convenient and automatized BAKS for most use cases
Typically, you will have many spike trains from many neurons that you want to optimize alpha and estimate the firing rate for, but processing each individually would be very slow.
`pyBAKS.autoBAKS()` is a convenience function that will take a list, series, or 2D array of spike trains and corresponding time vectors, and parallelize the optimization and firing rate estimation for each spike train. It returns a pandas series containing the optimized firing rate estimate for each spike train. 

**If you ever use pyBAKS in a pipeline, pyBAKS.autoBAKS() should be the only pyBAKS function you really need to get firing rates from spike trains.**

```python
import pyBAKS
#Example 1: inputing spike trains from a dataframe:
df = pyBAKS.BAKS_test.generate_sim_df() #generate a dataframe with 30 different 5s spike trains with 4 unique variations in firing rate for each, and corresponding time vectors
BAKS_rates = pyBAKS.autoBAKS(df['Spikes'], df['Time']) #optimize alpha and estimate firing rate for each spike train
df['BAKS_rates'] = BAKS_rates #add the firing rate estimates to the dataframe

#Example 2: inputing spike trains from a list of arrays:
SpikeList = df['Spikes'].tolist() #convert the spike train column to a 2D array
TimeList = df['Time'].tolist() #convert the time vector column to a 2D array
BAKS_rates = pyBAKS.autoBAKS(SpikeList, TimeList) #optimize alpha and estimate firing rate for each spike train

#Example 3: inputing spike trains from a 2D array:
SpikeArray = np.array(SpikeList) #convert the spike train column to a 2D array
TimeArray = np.array(TimeList) #convert the time vector column to a 2D array
BAKS_rates = pyBAKS.autoBAKS(SpikeArray, TimeArray) #optimize alpha and estimate firing rate for each spike train

```
You can also pass 1D spike trains and time arrays to `pyBAKS.autoBAKS()`. 

### Alternative functions: MISE optimization and rolling window smoothing

Ahmadi 2019 uses mean integrated squared error (MISE) to optimize alpha. We have included a function `pyBAKS.optimize_alpha_MISE()` that takes a spike train and time vector as input, samples the firing rate from 4 randomly placed and sized windows of time, and generates a spike train with 4 variations in firing rate, for however many iterations you specify.
Input spiking data is "rehashed" in this way, because a spike train with a known latent firing rate is required to calculate MISE, but "real" spike trains do not have a known latent firing rate, so we attempt to simulate spike trains with "realistic" firing rates by sampling from the input spike train.
Alpha is then optimized against these simulated spike trains. In our simulations evaulating this method, we find that optimizing alpha according to MISE as implemented here is slower and less accurate (as evaluated by both MISE and log-likelihood) than using MLE, but we have included it for the sake of completeness. 

It may sometimes be useful to double check if using a tool like BAKS on your data actually outperforms simpler methods. 
For this reason, we have included functions for rate estimation using a rolling window of time, rather than the adaptive kernel (`pyBAKS.rolling_window()`), as well as optimiztion of the window size using both MLE and MISE (`pyBAKS.get_optimized_rolling_rates_MLE()` and `pyBAKS.get_optimized_rolling_rates_MISE()`).

See pyBAKS.BAKS_test.BAKS_test_sim_data() for a demonstration comparing both BAKS and rolling window rate-estimation, as optimized by both MLE and MISE, on a simulated poisson spike train. 
