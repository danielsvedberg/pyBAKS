import pyBAKS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def generate_trial(Time, rates):
    dt = Time[1] - Time[0]
    Spikes = np.zeros(len(Time))
    Rates = np.zeros(len(Time))
    step = int(len(Time) / len(rates))

    for i in rates:
        epoch_start = rates.index(i) * step
        epoch_end = epoch_start + step
        prob = i * dt
        spikes = np.random.rand(step) <= prob
        Spikes[epoch_start:epoch_end] = spikes
        epoch_rates = np.ones(step) * i
        Rates[epoch_start:epoch_end] = epoch_rates

    Spikes = Spikes.astype(int)
    return Spikes, Rates

def sim_trials(n_trials=30, trial_length=5, n_epochs=4):
    # generate a 5000-bin time array with a dt of 0.001s--a 5-second recording
    Time = np.arange(0, trial_length, 0.001)

    #generate a random firing rate (below 70hz) for each of n_epochs
    rates = []
    for i in range(n_epochs):
        rate = np.random.rand() * 70
        rates.append(rate)

    #generate n-trials of spikes
    Spike_list = []
    Rate_list = []
    Time_list = []
    trial_id = []
    for i in range(n_trials):
        Spikes, Rates = generate_trial(Time, rates)
        Spike_list.append(Spikes)
        Rate_list.append(Rates)
        Time_list.append(Time)
        trial_id.append(i)

    df = pd.DataFrame(data={"trial_id": trial_id, "Spikes": Spike_list, "Rates": Rate_list, "Time": Time_list})

    return df

def sim_df(n_units=None, n_trials=None, trial_length=None, n_epochs=None):

    if n_units is None:
        n_units=10
    if n_trials is None:
        n_trials=30
    if trial_length is None:
        trial_length=5
    if n_epochs is None:
        n_epochs=4

    results = []
    for i in range(n_units):
        unit_df = sim_trials(n_trials, trial_length, n_epochs)
        unit_df['unitID'] = i
        results.append(unit_df)

    df = pd.concat(results)
    return df

def test_sim_df():
    df = sim_df(n_trials=10,n_units=2)
    print(df.head())
    test = df.loc[df['unitID'] == 0]
    testdf, testfr, testalpha = pyBAKS.optimize_alpha_MLE(test['Spikes'], test['Time'])

    full_res, best_res = pyBAKS.dfBAKS(df, 'Spikes', 'Time', 'unitID')
    alpha_check = full_res.groupby(['alpha', 'unitID'], as_index=False)['log_likelihood'].mean()
    ba_check = alpha_check.groupby('unitID')['log_likelihood'].idxmax()
    ba_chec = alpha_check.loc[ba_check]

    g = sns.relplot(x="alpha", y="log_likelihood", hue="unitID", data=full_res, kind="line")
    plt.show()
def test_sim_data():
    df = sim_trials(n_trials=1)
    Spikes = np.array(df['Spikes'].tolist()).flatten()
    Rates = np.array(df['Rates'].tolist()).flatten()
    Time = np.array(df['Time'].tolist()).flatten()
    # generate a rolling-window average of the test data for comparison
    winRate_MISE, _, _ = pyBAKS.get_optimized_rolling_rates_MISE(Spikes, Time, nIter=30)
    _, _, winRate_MLE = pyBAKS.optimize_window_MLE(Spikes, Time)

    BAKSrate_MISE, h, ba_MISE = pyBAKS.get_optimized_BAKSrates_MISE(Spikes, Time, nIter=30)
    OAdf, BAKSrate_MLE, ba_MLE = pyBAKS.optimize_alpha_MLE(Spikes, Time)

    pyBAKS.plot_spike_train_vs_BAKS_vs_rolling(Spikes, Rates, BAKSrate_MISE, winRate_MISE.flatten(), Time)
    pyBAKS.plot_spike_train_vs_BAKS_vs_rolling(Spikes, Rates, BAKSrate_MLE, winRate_MLE.flatten(), Time)

    win_MISE_MISE = pyBAKS.getMISE(Rates, winRate_MISE)
    win_MLE_MISE = pyBAKS.getMISE(Rates, winRate_MLE)
    BAKS_MISE_MISE = pyBAKS.getMISE(Rates, BAKSrate_MISE)
    BAKS_MLE_MISE = pyBAKS.getMISE(Rates, BAKSrate_MLE)

    win_MISE_LL = pyBAKS.firingrate_loglike(Spikes, winRate_MISE)
    win_MLE_LL = pyBAKS.firingrate_loglike(Spikes, winRate_MLE)
    BAKS_MISE_LL = pyBAKS.firingrate_loglike(Spikes, BAKSrate_MISE)
    BAKS_MLE_LL = pyBAKS.firingrate_loglike(Rates, BAKSrate_MLE)

    #make a pandas dataframe of the results
    smoothingtype = ["rolling_window", "rolling_window", "pyBAKS", "pyBAKS"]
    optimizationtype = ["sim_MISE", "MLE", "sim_MISE", "MLE"]
    MISEs = [win_MISE_MISE, win_MLE_MISE, BAKS_MISE_MISE, BAKS_MLE_MISE]
    LLs = [win_MISE_LL, win_MLE_LL, BAKS_MISE_LL, BAKS_MLE_LL]

    df = pd.DataFrame(data={"smoothing_method": smoothingtype, "optimization_method": optimizationtype, "MISE": MISEs, "log_likelihood": LLs})
    print(df)

def test_autoBAKS():
    print("testing autoBAKS on simulated array of single-unit")
    Spikes, Rates, Time = sim_trials()
    BAKSrate = pyBAKS.autoBAKS(Spikes, Time)
    if BAKSrate is None:
        print("autoBAKS failed to fit array")
    else:
        print("autoBAKS array fit success")

    print("testing autoBAKS on simulated dataframe with multiple units")
    df = sim_trials()
    df['BAKSrate'] = pyBAKS.autoBAKS(df['Spikes'], df['Time'])
    if df['BAKSrate'].isnull().values.any():
        print("autoBAKS failed to fit all units")
    else:
        print("autoBAKS dataframe fit success")

    print("testing autoBAKS on simulated list of numpy arrays for multiple units")
    Spikes = df['Spikes'].tolist()
    Time = df['Time'].tolist()
    BAKSrate = pyBAKS.autoBAKS(Spikes, Time)
    if BAKSrate is None:
        print("autoBAKS failed list of numpy arrays test")
    else:
        print("autoBAKS list of numpy arrays test success")

    print("testing autoBAKS on simulated 2D numpy array for multiple units")
    Spikes = np.array(df['Spikes'].tolist())
    Time = np.array(df['Time'].tolist())
    BAKSrate = pyBAKS.autoBAKS(Spikes, Time)
    if BAKSrate is None:
        print("autoBAKS failed 2D numpy array test")
    else:
        print("autoBAKS 2D numpy array test success")






