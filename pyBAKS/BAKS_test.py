import pyBAKS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from inspect import getsourcefile
from os.path import abspath
def generate_trial(Time, rates):
    dt = Time[1] - Time[0]
    Spikes = np.zeros(len(Time))
    Rates = np.zeros(len(Time))
    steps = np.random.randint(0, len(Time), len(rates) - 1)
    steps = np.append(steps, [0, len(Time)])
    steps = np.sort(steps)
    #step = int(len(Time) / len(rates))

    for idx, i in enumerate(rates):
        epoch_start = steps[idx]
        epoch_end = steps[idx + 1]
        step = int(epoch_end - epoch_start)
        prob = i * dt
        spikes = np.random.rand(step) <= prob
        Spikes[epoch_start:epoch_end] = spikes
        epoch_rates = np.ones(step) * i
        Rates[epoch_start:epoch_end] = epoch_rates

    Spikes = Spikes.astype(int)
    return Spikes, Rates

def sim_trial_array(n_trials=30, trial_length=5, n_epochs=4, dt=0.001):
    Time = np.arange(0, trial_length, dt)

    # generate a random firing rate (below 70hz) for each of n_epochs
    rates = []
    for i in range(n_epochs):
        rate = np.random.rand() * 70
        rates.append(rate)

    # generate n-trials of spikes
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
    #convert to numpy arrays
    Spikes = np.array(Spike_list)
    Rates = np.array(Rate_list)
    Times = np.array(Time_list)
    trialIDs = np.array(trial_id)
    return Spikes, Rates, Times, trialIDs


def sim_trial_df(n_trials=30, trial_length=5, n_epochs=4, dt=0.001):
    # generate a 5000-bin time array with a dt of 0.001s--a 5-second recording
    Spike_list, Rate_list, Time_list, trial_id = sim_trial_array(n_trials, trial_length, n_epochs, dt)
    #convert Spike_list, Rate_list, and Time_list from trials x time to a list of time arrays
    Spike_list = Spike_list.tolist()
    Rate_list = Rate_list.tolist()
    Time_list = Time_list.tolist()
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
        unit_df = sim_trial_df(n_trials, trial_length, n_epochs)
        unit_df['unitID'] = i
        results.append(unit_df)

    df = pd.concat(results)
    return df

def test_sim_data():
    #get the directory where the file is
    path = abspath(getsourcefile(lambda: 0))
    #get everything up to and including "pyBAKS"
    path = path.split("pyBAKS")[0]
    save_dir = path + "pyBAKS/"
    ntrials = 5
    spikearr, ratearr, timearr, trialarr = sim_trial_array(n_trials=ntrials, trial_length=5, n_epochs=4, dt=0.001)
    # generate a rolling-window average of the test data as a baseline
    df, best_window_size = pyBAKS.optimize_window_MISE(spikearr, timearr, 10)
    #plot MISE vs window size
    MISE = df['MISE']
    best_MISE = df['MISE'].min()
    window_size = df['window_size']
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    sns.lineplot(x=window_size, y=MISE, ax=axs)
    axs.axvline(x=best_window_size, color='red', linestyle='--')
    title = "Rolling Window: MISE vs Window Size (s)\nBest Window Size: " + str(best_window_size) + ", MISE: " + str(best_MISE)
    plt.title(title)
    plt.savefig(save_dir + "rolling_window_MISE.png")
    plt.show()

    winRate_MISE, _, _ = pyBAKS.get_optimized_rolling_rates_MISE(spikearr, timearr, nIter=30)
    rollingMISE = pyBAKS.getMISE(ratearr, winRate_MISE)
    def plot_sim_vs_smooth(smootharr, flag):
        #plot winRate_Mise against the true rates
        ntrials = ratearr.shape[0]
        fig, axs = plt.subplots(ntrials, 2, figsize=(10, ntrials*1.5), sharex=True)
        for i, ax in enumerate(axs):
            if i == ntrials:
                pass
            else:
                #plot the latent rate first:
                ax[1].plot(timearr[i], ratearr[i], label="Latent Rate", color='red')
                ax[1].plot(timearr[i], smootharr[i], label="Smoothed Rate", color='black')
                ax[1].set_ylabel("Hz")
                ax2 = ax[1].twinx()
                ax2.set_ylabel("trial " + str(i))
                ax2.set_yticks([])
                ax[1].legend()
                ax[0].plot(timearr[i], spikearr[i], color='Black')
                ax[0].set_ylabel('spikes')
        ax = axs[-1]
        ax[1].set_xlabel("Time (ms)")
        ax[0].set_xlabel("Time (ms)")

        pltnm = "simulated data vs " + flag
        fig.suptitle(pltnm)
        plt.tight_layout()
        plt.show()
        fig.savefig(save_dir + pltnm + ".png")

    flag = 'rolling window smoothing, MISE: ' + str(rollingMISE)
    plot_sim_vs_smooth(winRate_MISE, flag=flag)

    df, best_alpha = pyBAKS.optimize_alpha_MISE(spikearr, timearr, 10)
    #plot MISE vs alpha
    MISE = df['MISE']
    best_MISE = df['MISE'].min()
    alpha = df['alpha']
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    sns.lineplot(x=alpha, y=MISE, ax=axs)
    axs.axvline(x=best_alpha, color='red', linestyle='--')
    nm = "pyBAKS: MISE vs Alpha\nBest Alpha: "
    title = nm + str(best_alpha) + ", MISE: " + str(best_MISE)
    plt.title(title)
    plt.show()
    plt.savefig(save_dir + "pyBAKS_MISE_vs_alpha.png")

    BAKSrate_MISE, h, ba_MISE = pyBAKS.get_optimized_BAKSrates_MISE(spikearr, timearr, nIter=10)
    BAKSMISE = pyBAKS.getMISE(ratearr, BAKSrate_MISE)
    flag = 'BAKS smoothing, MISE: ' + str(BAKSMISE)
    plot_sim_vs_smooth(BAKSrate_MISE, flag=flag)