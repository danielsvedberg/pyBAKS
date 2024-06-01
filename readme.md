# pyBAKS: python implementation of Bayesian Adaptive Kernel Smoother (BAKS) for estimation of neuronal firing rate

Daniel Svedberg <br> Brandeis University (2023)

Based on the following work by Nur Ahmadi: <br>
github.com/nurahmadi/BAKS/ <br>
github.com/nurahmadi/spike_bmi [spike_bmi.bmi.baks()]<br>
Ahmadi, Nur, Timothy G. Constandinou, and Christos-Savvas Bouganis. "Estimation of neuronal firing rate using Bayesian Adaptive Kernel Smoother (BAKS)." Plos one 13.11 (2018): e0206794

## Introduction
BAKS is a method for estimating the firing rate of a neuron from a spike train. 
It is based on a Bayesian framework and uses an adaptive-width kernel to smooth the spike train. 
The method is described in detail in the paper cited above.<br>
pyBAKS is a python implementation of the original BAKS algorithm written by Nur Ahmadi, 
with added convenience functions for automatically handling various data structures, 
optimizing the alpha parameter, and parallel-processing of multiple spike arrays simultaneously.<br>
Also included in pyBAKS is the testing module pyBAKS.BAKS_test, 
which contains demos for pyBAKS. 

## Installation & Loading
you can install pyBAKS via pip:
```bash
pip install pyBAKS
```
You can then import pyBAKS from within a python console or script:
```python
import pyBAKS
```
## Usage

### Core function: pyBAKS.baks()
The core function of pyBAKS is `pyBAKS.baks()`. It takes a 1D vector of spiketimes, and the alpha parameter as input, and returns a firing rate estimate, and the kernel bandwidth for each index: 
```python  
import pyBAKS
import numpy as np

spike_times = [0.1, 0.3, 0.5]
time = np.arange(0, 1, 0.01)
alpha = 4
BAKS_firing_rate, kernel_bandwidth = pyBAKS.baks(spike_times, time, alpha)
```
### Parameter tuning
Because alpha is a parameter that must be tuned, the function `pyBAKS.optimize_alpha_MISE()` is included, 
which optimizes the parameter alpha for a 2D matrix of spikes

### Demo of pyBAKS Parameter tuning

```python
#First, we generate some simulated data
#this data generates 5 5s trials of poisson spiking, 
#each trial contains 3 modulations through the same sequence of 4 firing rates
#but the timing of each modulation is random
ntrials = 5
spikearr, ratearr, timearr, trialarr = pyBAKS.BAKS_test.sim_trial_array(n_trials=ntrials, trial_length=5, n_epochs=4, dt=0.001)
#spikearr is a 2D spike array [trials * bins]
#ratearr is the (latent) rate array [trials * bins]
#timearr is the time array [trials * bins]
#trialarr is the trial index [trials]

#then you can pass your spike array and time array through the convenience function
#to automatically optimize the alpha and then extract the optimized BAKS rates
BAKSrate_MISE, h, ba_MISE = pyBAKS.get_optimized_BAKSrates_MISE(spikearr, timearr, nIter=10)
#BAKSrate_MISE is the rate matrix [trials * bins]
#h is the scale parameter
#ba_MISE is the alpha value chosen by the algorithm
```