import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from scipy.special import factorial
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed


def BAKS(SpikeTimes, Time, a, b=None):
    if np.all(np.isin(SpikeTimes, [0, 1])) and len(
            SpikeTimes) > 2:  # if SpikeTimes is a spike array, convert to spike times
        if len(SpikeTimes) == 0 or SpikeTimes is None:
            print("warning: SpikeTimes is empty, returning zero arrays")
            FiringRate = np.zeros(len(Time))
            h = np.zeros(len(Time))
            return FiringRate, h
        else:
            print("SpikeTimes is a spike array, converting to spike times")
            if len(SpikeTimes) != len(Time):
                raise ValueError("Spike array and time array must have the same length.")
            SpikeTimes = extract_spike_times(SpikeTimes, Time)

    if a < 1:
        raise ValueError("according to Ahmadi et al., alpha (a) must be >= 1")

    N = len(SpikeTimes)
    sumnum = 0
    sumdenum = 0

    if b is None:  # if b is not specified, use the default calculation from Ahmadi et al.
        b = N ** (4 / 5)

    # calculate h (eq 11 in Ahmadi et al.)
    for i in range(N):
        innerterm = ((((Time - SpikeTimes[i]) ** 2) / 2) + (1 / b))
        numerator = innerterm ** (-a)
        denumerator = innerterm ** (-a - 0.5)
        sumnum += numerator
        sumdenum += denumerator
    h = (gamma(a) * sumnum) / (gamma(a + 0.5) * sumdenum)

    # calculate firing rate (eq 12 in Ahmadi et al.)
    FiringRate = np.zeros(len(Time))
    for j in range(N):
        K = (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-((Time - SpikeTimes[j]) ** 2) / (2 * h ** 2))
        FiringRate += K

    return FiringRate, h


def rolling_window(Spikes, dt, ws):
    """
    rolling_window is used to calculate the rolling window average of a spike array
    :param Spikes: 1D binary array of spike emissions
    :param dt: length of the time step
    :param ws: window size in seconds
    :return: winAvg: 1D array of the rolling window average of Spikes
    """

    if Spikes.ndim == 1:
        Spikes = Spikes.reshape(1, -1)

    window = ws / dt
    window = int(window)
    kernel = np.ones(window) / window

    result = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=1, arr=Spikes)
    winAvg = result / dt

    ll = firingrate_loglike(Spikes, winAvg)
    return winAvg, ll


def firingrate_loglike(Spikes, FiringRate):
    if Spikes.ndim == 2:
        loglike = np.sum(np.log((FiringRate ** Spikes * np.exp(-FiringRate)) / factorial(Spikes)), axis=1)
        if len(loglike) == 1:
            loglike = loglike[0]
    else:
        loglike = np.sum(np.log((FiringRate ** Spikes * np.exp(-FiringRate)) / factorial(Spikes)))
    return loglike


def extract_spike_times(Spikes, Time):
    """
    :param Spikes: 1D binary emissions array
    :param Time: 1D time array same length as Spikes
    :return: SpikeTimes: 1D array of spike times
    """

    if len(Spikes) != len(Time):
        raise ValueError("Spike array and time array must have the same length.")
    spikeIdxs = np.where(Spikes == 1)
    SpikeTimes = Time[spikeIdxs]

    return SpikeTimes


def getMISE(true_rate, est_rate):
    """
    getMISE is used to calculate the mean integrated square error (MISE) between the true rate and estimated rate.
    to use this tool, you should first generate a simulated "known" rate array, then use that to generate
    a simulated spike train, and then get the BAkS-estimated rate from the simulated spike train.
    getMISE calculates the MISE between an a known-rate array and a BAKS-estimated rate array.
    :param true_rate: a rate array with the (known) true rate, can be generated by spikearray_poissonsim
    :param est_rate: an array of the estimated rate
    :return MISE: mean integrated squared error between true rate and estimated rate
    """
    nt = len(true_rate)
    MISE = np.sum((est_rate - true_rate) ** 2) / nt
    return MISE


def spikearray_poissonsim(Spikes, Time):
    """
    a simulated spike array with known firing rate is required to optimize alpha for BAKS
    this function generates a similuated spike array based on Spikes
    :param Spikes: spike array of real spikes from which to generate simulated spikes
    :param Time: time array for Spikes
    :param nIter: number of iterations to run
    :return sim_spikes: simulated spike array
    :return sim_rate: simulated rate array
            the first quarter is 1/4 the rate, the second quarter is 1/2 the rate,
            the 3rd quarter is the rate, and the 4th quarter is 2x the rate
    :return sim_time: time array for sim_spikes
    """

    if Spikes.ndim == 1:
        Spikes = Spikes.reshape(1, -1)
        Time = Time.reshape(1, -1)

    n_time_bins = Spikes.shape[1]
    time_range = Time[0, -1] - Time[0, 0]

    dt = time_range / n_time_bins

    sim_spikes = np.zeros(Spikes.shape)
    sim_rate = np.zeros(Spikes.shape)
    step = int(n_time_bins / 4)
    ntrls = Spikes.shape[0]

    for i in range(4):
        samp = np.random.randint(0, n_time_bins, size=2)
        start = np.min(samp)
        end = np.max(samp)

        slice = Spikes[:, start:end]
        slicerate = np.sum(slice, axis=1) / ((end - start) * dt)
        sliceprob = slicerate / n_time_bins
        sliceprob = sliceprob.reshape(-1, 1)
        slice_vals = np.random.rand(ntrls, step)
        slice_spikes = slice_vals <= sliceprob
        sim_spikes[:, step * i:step * (i + 1)] = slice_spikes
        sim_rate[:, step * i:step * (i + 1)] = slicerate.reshape(-1, 1)

    sim_time = Time
    return sim_spikes, sim_rate, sim_time


def optimize_alpha_MISE(Spikes, Time, nIter=100, alpha_start=1, alpha_end=10, alpha_step=0.1):
    """
    optimize_alpha is used to optimize alpha for BAKS from a 1D array of real spiking data.
    it uses the real data to generate a simulated spike array with known rate,
    then uses that to generate a BAKS-estimated rate,
    then getMISE to calculate the MISE between the known rate and the BAKS-estimated rate,
    and repeats this over a range of alpha values to find the alpha value with the lowest MISE.
    :param Spikes: 1D spike train from real data
    :param Time: time array corresponding to Spikes
    :param nIter: how many iterations to run
    :param alpha_start: minimum alpha value to test.
    :param alpha_end: maximum alpha value to test.
    :param alpha_step: step size for alpha values.
    :return: df: pandas dataframe with iternums, MISEs, and alphas
    :return: best_alpha: alpha value with lowest MISE
    """

    alpha_range = np.arange(alpha_start, alpha_end, alpha_step)
    iternums = []
    MISEs = []
    alphas = []
    likelihoods = []
    for iter in range(nIter):
        sim_spikes, sim_rate, sim_time = spikearray_poissonsim(Spikes, Time)
        sim_spiketimes = extract_spike_times(sim_spikes, sim_time)
        for a in alpha_range:
            BAKSrate, h, = BAKS(sim_spiketimes, Time, a)
            lh = firingrate_loglike(sim_spikes, BAKSrate)
            MISE = getMISE(sim_rate, BAKSrate)
            iternums.append(iter)
            MISEs.append(MISE)
            alphas.append(a)
            likelihoods.append(lh)

    # make a pandas table with iternums, MISEs, and alphas
    df = pd.DataFrame({'iteration': iternums, 'MISE': MISEs, 'alpha': alphas, 'likelihood': likelihoods})
    df_avg = df.groupby(['alpha']).mean()
    # get alpha value with lowest MISE
    best_alpha = df_avg['MISE'].idxmin()
    if df_avg['MISE'].iloc[-1] == best_alpha:
        print("Warning: lowest MISE is at the end of the range of alpha values tested. "
              "Consider increasing alpha_end.")

    return df, best_alpha


def parse_dims(Spikes):
    ndim = None
    kind = None
    if isinstance(Spikes, pd.Series):
        kind = 'series'
        # determine if Spikes is a list of arrays or a single array
        if isinstance(Spikes.iloc[0], np.ndarray):
            ndim = 2
        else:
            ndim = 1
    elif isinstance(Spikes, list):
        kind = 'list'
        # determine if Spikes is a list of arrays or a single array
        if isinstance(Spikes[0], np.ndarray):
            ndim = 2
        else:
            ndim = 1
    elif isinstance(Spikes, np.ndarray):
        kind = 'numpy'
        ndim = Spikes.ndim

    if ndim is None or kind is None:
        raise ValueError("Spikes is not a list, array, or pandas series")
    else:
        return ndim, kind


def optimize_alpha_MLE(Spikes, Time, alpha_start=1, alpha_end=10.1, alpha_step=0.1, ndim=None, kind=None, unitID=None, output_df=True):
    # generate alphas to be optimized over
    alpha_range = np.arange(alpha_start, alpha_end, alpha_step)
    df = None
    best_alpha = None
    best_FiringRate = None
    alphas = []
    loglikes = []
    FiringRates = []
    bandwidths = []

    def get_BAKS():
        spiketimes = extract_spike_times(spk, tm)
        for a in alpha_range:
            BAKSrate, h, = BAKS(spiketimes, tm, a)
            ll = firingrate_loglike(spk, BAKSrate)
            alphas.append(a)
            loglikes.append(ll)
            FiringRates.append(BAKSrate)
            bandwidths.append(h)

    # determine if Spikes is a list of arrays or a single array
    if ndim or kind is None:
        ndim, kind = parse_dims(Spikes)

    if ndim == 1:
        spk = Spikes
        tm = Time
        get_BAKS()

        # make a pandas table with iternums, MISEs, and alphas
        df = pd.DataFrame(
            {'BAKSrate': FiringRates, 'bandwidth': bandwidths, 'log_likelihood': loglikes, 'alpha': alphas})
        # get alpha value with highest log likelihood
        bestidx = df['log_likelihood'].idxmax()
        best_alpha = df['alpha'].iloc[bestidx]

        # get the firing rate with the highest log likelihood
        best_FiringRate = FiringRates[bestidx]
        if unitID is not None:
            df['unitID'] = unitID

    elif ndim == 2:
        if len(Spikes) == len(Time):
            for spk, tm in zip(Spikes, Time):
                get_BAKS()
        elif len(Spikes[0]) == len(Time):
            tm = Time
            for spk in Spikes:
                get_BAKS()

        # make a dataframe
        df = pd.DataFrame(
            {'BAKSrate': FiringRates, 'log_likelihood': loglikes, 'alpha': alphas})
        # calculate average log-likelihood for each alpha, get the best alpha
        best_alpha = df.groupby(['alpha'])['log_likelihood'].mean().idxmax()

        # get rows from df where alpha == best_alpha
        best_FiringRate = df[df['alpha'] == best_alpha]['BAKSrate']
        if unitID is not None:
            df['unitID'] = unitID

    if kind == 'numpy':
        best_FiringRate = best_FiringRate.to_numpy()
        best_FiringRate = np.vstack(best_FiringRate)
    elif kind == 'list':
        best_FiringRate = best_FiringRate.to_list()
    elif kind == 'series':
        action = "do nothing"
    else:
        raise ValueError("Spikes is not a list, array, or pandas series")

    if output_df:
        return df, best_FiringRate, best_alpha
    else:
        del df
        return best_FiringRate, best_alpha

def optimize_window_MLE(Spikes, Time, ws_start=0.1, ws_end=5, ws_step=0.1):
    ws_range = np.arange(ws_start, ws_end, ws_step)

    dt = Time[1] - Time[0]
    loglikes = []
    FiringRates = []

    for ws in ws_range:
        winAvg, ll = rolling_window(Spikes, dt, ws)
        loglikes.append(ll)
        FiringRates.append(winAvg)

    df = pd.DataFrame({'window_size': ws_range, 'log_likelihood': loglikes})
    bestidx = df['log_likelihood'].idxmax()
    best_window_size = df['window_size'].iloc[bestidx]
    best_FiringRate = FiringRates[bestidx]
    return df, best_window_size, best_FiringRate


def optimize_window_MISE(Spikes, Time, nIter=100, ws_start=0.1, ws_end=None, ws_step=0.1):
    """
    optimize_alpha is used to optimize alpha for BAKS from a 1D array of real spiking data.
    it uses the real data to generate a simulated spike array with known rate,
    then uses that to generate a BAKS-estimated rate,
    then getMISE to calculate the MISE between the known rate and the BAKS-estimated rate,
    and repeats this over a range of alpha values to find the alpha value with the lowest MISE.
    :param Spikes: 1D spike train from real data
    :param Time: time array corresponding to Spikes
    :param nIter: how many iterations to run
    :param alpha_start: minimum alpha value to test.
    :param alpha_end: maximum alpha value to test.
    :param alpha_step: step size for alpha values.
    :return: df: pandas dataframe with iternums, MISEs, and alphas
    :return: best_alpha: alpha value with lowest MISE
    """

    if Spikes.ndim == 1:
        Spikes = Spikes.reshape(1, -1)
        Time = Time.reshape(1, -1)

    if ws_end is None:
        ws_end = Time[0, -1] - Time[0, 0]

    dt = Time[0, 1] - Time[0, 0]
    window_range = np.arange(ws_start, ws_end, ws_step)

    iternums = []
    MISEs = []
    windows = []
    likelihoods = []

    for iter in range(nIter):
        sim_spikes, sim_rate, sim_time = spikearray_poissonsim(Spikes, Time)
        for ws in window_range:
            rolling_rate, _ = rolling_window(sim_spikes, dt, ws)
            lh = firingrate_loglike(sim_spikes, rolling_rate)
            MISE = getMISE(sim_rate, rolling_rate)
            iternums.append(iter)
            MISEs.append(MISE)
            windows.append(ws)
            likelihoods.append(lh)

    # make a pandas table with iternums, MISEs, and alphas
    df = pd.DataFrame({'iteration': iternums, 'MISE': MISEs, 'window_size': windows})
    df_avg = df.groupby('window_size').mean()
    # get alpha value with lowest MISE
    best_window = df_avg['MISE'].idxmin()
    if df_avg['MISE'].iloc[-1] == best_window:
        print("Warning: lowest MISE is at the end of the range of alpha values tested. "
              "Consider increasing alpha_end.")

    return df, best_window


def get_optimized_BAKSrates_MISE(Spikes, Time, nIter=10):
    df, best_alpha = optimize_alpha_MISE(Spikes, Time, nIter)
    SpikeTimes = extract_spike_times(Spikes, Time)
    BAKSrate, h = BAKS(SpikeTimes, Time, a=best_alpha)

    return BAKSrate, h, best_alpha


def get_optimized_rolling_rates_MISE(Spikes, Time, nIter=10):
    df, best_window_size = optimize_window_MISE(Spikes, Time, nIter)
    if Spikes.ndim == 2:
        dt = Time[0, 1] - Time[0, 0]
    else:
        dt = Time[1] - Time[0]

    winAvg, ll = rolling_window(Spikes, dt, best_window_size)

    return winAvg, ll, best_window_size


def plot_MISE_v_alpha(df, best_alpha):
    """
    plot_MISE_v_alpha is used to plot the MISE values from optimize_alpha
    :param df: pandas dataframe from optimize_alpha
    :return: plot
    """
    sns.lineplot(data=df, x='alpha', y='MISE')
    # plot a vertical line at best_alpha
    plt.axvline(x=best_alpha, color='r', linestyle='--')
    plt.show()


def plot_spike_train_vs_BAKS_vs_rolling(Spikes, TrueRate, BAKSrate, winAvg, Time):
    """
    plot_spike_train_vs_BAKS is used to plot the spike train vs. the BAKS-estimated rate
    :param SpikeTimes: spike times from real data
    :param Time: time array corresponding to SpikeTimes
    :param BAKSrate: BAKS-estimated rate
    :return: plot
    """
    # make a tiled plot, upper tile is spike train, lower tile is BAKS-estimated rate
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
    ax1.plot(Time, Spikes)
    ax2.plot(Time, TrueRate)
    ax3.plot(Time, BAKSrate)
    ax4.plot(Time, winAvg)
    plt.show()


def parallel_apply(Spikes, Time, func, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(delayed(func)(x, y) for x, y in zip(Spikes, Time))
    return pd.Series(results)


def dfBAKS(df, spikes_col, time_col, idxcols=None, n_jobs=-1):
    """
    dfBAKS is used to apply BAKS to a pandas dataframe of spike trains
    :param df: pandas dataframe
    :param spikes_col: name of column containing spike trains
    :param time_col: name of column containing time arrays
    :param idxcols: list of column names to use as index for the output dataframe
    :return: df: pandas dataframe with BAKSrate column added
    """
    df = df.copy().reset_index(drop=True)

    if n_jobs == 0:
        full_df = []
        best_df = []
        for key, group in df.groupby(idxcols):
            res_df, fr, alpha = optimize_alpha_MLE(group[spikes_col], group[time_col])
            full_df.append(res_df)
            best_df.append(res_df.loc[res_df['alpha'] == alpha])

        full_df = pd.concat(full_df).reset_index(drop=True)
        best_df = pd.concat(best_df).reset_index(drop=True)
        df[['BAKSrate', 'bandwidth', 'log_likelihood', 'alpha']] = best_df[
            ['BAKSrate', 'bandwidth', 'log_likelihood', 'alpha']]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(optimize_alpha_MLE)(group[spikes_col], group[time_col], unitID=key) for key, group in
            df.groupby(idxcols))

        full_df = []
        best_df = []
        for res_df, fr, best_alpha in results:
            print(best_alpha)
            full_df.append(res_df)
            best_df.append(res_df.loc[res_df['alpha'] == best_alpha])

        full_df = pd.concat(full_df).reset_index(drop=True)
        best_df = pd.concat(best_df).reset_index(drop=True)
        df = df.reset_index(drop=True)
        df[['BAKSrate', 'bandwidth', 'log_likelihood', 'alpha']] = best_df[
            ['BAKSrate', 'bandwidth', 'log_likelihood', 'alpha']]
        # retrieve the rows of full_df, where for each  alpha == best_alpha for each idxcol

    return full_df, df


def autoBAKS(Spikes, Time, ndim=None, unit_index=None):
    """
    autoBAKS is used to automatically optimize BAKS parameter alpha and generate BAKS-smoothed firing rates
    for various types of input data. As long as Spikes and Time are the same data type and size and of a supported data
    type, autoBAKS will return a BAKS-smoothed firing rate.
    Both Spikes and time can be:
    (1D) a list, a pandas series, or a numpy array.
    (2D) a list of arrays, a pandas series of arrays, or a 2D numpy array.
    :param Spikes: must be contain binary emissions (spikes)
    :param Time: must contain time values corresponding to Spikes
    :return: BAKSrate: BAKS-smoothed firing rate
    """

    # detect if Spikes is a list of arrays, a series, or a single array
    if unit_index is not None:
        if ndim == 1:
            raise ValueError("ndim must be 2 or None if unit_index is not None,"
                             "since unit_index implies existence of multiple units")

        df = pd.DataFrame({'Spikes': Spikes, 'Time': Time, 'unitID': unit_index})
        df = dfBAKS(df, 'Spikes', 'Time', 'unitID')
        return df['BAKSrate']

    elif ndim is None:
        if isinstance(Spikes, list):
            # detect is Spikes contains binary data. if not, raise error
            if not np.all(np.isin(Spikes, [0, 1])):
                raise ValueError("Spikes must be a binary array of spike emissions")

            print("Spikes is a list")
            if isinstance(Spikes, list) and all(isinstance(item, np.ndarray) for item in Spikes):
                print("Spikes is a list of arrays")
                print("warning, autoBAKS on a list of arrays is not yet well-tested.")
                ndim = 2
            else:
                print("Spikes is a list of non-arrays")
                print("warning, autoBAKS on a list of non-arrays is not yet well-tested.")
                ndim = 1

        elif isinstance(Spikes, pd.Series):
            print("Spikes is a pandas series")
            # detect if series is a list of arrays or a single array
            if isinstance(Spikes.iloc[0], np.ndarray):
                print("Spikes is a series of arrays, parallelizing")
                ndim = 2
            else:
                ndim = 1
        elif isinstance(Spikes, np.ndarray):
            print("Spikes is an array")
            if Spikes.ndim == 2:
                print("Spikes is a 2D array, parallelizing")
                print("warning, autoBAKS on a 2D array is not yet tested.")
                ndim = 2
            elif Spikes.ndim == 1:
                print("Spikes is a 1D array")
                ndim = 1
            else:
                print("Spikes is > 2 dimensions, only 1 and 2D arrays are supported")
                return None
        else:
            print("Spikes datatype not recognized. Please use a list, pandas series, or numpy array.")
            return None

    if ndim == 1:
        _, _, BAKSrate = optimize_alpha_MLE(Spikes, Time)
        return BAKSrate
    elif ndim == 2:
        results = parallel_apply(Spikes, Time, optimize_alpha_MLE, n_jobs=-1)
        _, _, BAKSrate = zip(*results)
        return BAKSrate
