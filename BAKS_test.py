import BAKS
import numpy as np
import pandas as pd
def generate_test_data():
    # generate a 1000-bin time array with a dt of 0.001s--a 1-second recording
    Time = np.arange(0, 5, 0.001)
    dt = Time[1] - Time[0]
    # generate a 1000-bin spike array with a spike probability that varies from 0 to 0.1, (between 0-100Hz), changing every 250 bins
    rates = []
    probs = []
    for i in range(4):
        rate = np.random.rand() * 70
        prob = rate * dt
        rates.append(rate)
        probs.append(prob)

    step = int(len(Time) / 4)
    epochs = [[0, step], [step, step * 2], [step * 2, step * 3], [step * 3, step * 4]]
    Spikes = np.zeros(len(Time))
    Rates = np.zeros(len(Time))
    for i in epochs:
        spikes = np.random.rand(step) <= probs[epochs.index(i)]
        Spikes[i[0]:i[1]] = spikes
        epoch_rates = np.ones(step) * rates[epochs.index(i)]
        Rates[i[0]:i[1]] = epoch_rates

    Spikes = Spikes.astype(int)
    return Spikes, Rates, Time

def generate_sim_df(nIter=30):
    results = []
    for i in range(nIter):
        Spikes, Rates, Time = generate_test_data()
        results.append({"unitID": i, "Spikes": Spikes, "Rates": Rates, "Time": Time})

    df = pd.DataFrame(results)
    return df

def test_sim_data():
    Spikes, Rates, Time = generate_test_data()
    # generate a rolling-window average of the test data for comparison
    winRate_MISE, _, _ = BAKS.get_optimized_rolling_rates_MISE(Spikes, Time, nIter=30)
    _, _, winRate_MLE = BAKS.optimize_window_MLE(Spikes, Time)

    BAKSrate_MISE, h, ba_MISE = BAKS.get_optimized_BAKSrates_MISE(Spikes, Time, nIter=30)
    df, ba_MLE, BAKSrate_MLE = BAKS.optimize_alpha_MLE(Spikes, Time)

    BAKS.plot_spike_train_vs_BAKS(Spikes, Rates, BAKSrate_MISE, winRate_MISE, Time)
    BAKS.plot_spike_train_vs_BAKS(Spikes, Rates, BAKSrate_MLE, winRate_MLE, Time)

    win_MISE_MISE = BAKS.getMISE(Rates, winRate_MISE)
    win_MLE_MISE = BAKS.getMISE(Rates, winRate_MLE)
    BAKS_MISE_MISE = BAKS.getMISE(Rates, BAKSrate_MISE)
    BAKS_MLE_MISE = BAKS.getMISE(Rates, BAKSrate_MLE)

    win_MISE_LL = BAKS.firingrate_loglike(Spikes, winRate_MISE)
    win_MLE_LL = BAKS.firingrate_loglike(Spikes, winRate_MLE)
    BAKS_MISE_LL = BAKS.firingrate_loglike(Spikes, BAKSrate_MISE)
    BAKS_MLE_LL = BAKS.firingrate_loglike(Rates, BAKSrate_MLE)

    #make a pandas dataframe of the results
    smoothingtype = ["rolling_window", "rolling_window", "BAKS", "BAKS"]
    optimizationtype = ["sim_MISE", "MLE", "sim_MISE", "MLE"]
    MISEs = [win_MISE_MISE, win_MLE_MISE, BAKS_MISE_MISE, BAKS_MLE_MISE]
    LLs = [win_MISE_LL, win_MLE_LL, BAKS_MISE_LL, BAKS_MLE_LL]

    df = pd.DataFrame(data={"smoothing_method": smoothingtype, "optimization_method": optimizationtype, "MISE": MISEs, "log_likelihood": LLs})
    print(df)

def test_autoBAKS():
    print("testing autoBAKS on simulated array of single-unit")
    Spikes, Rates, Time = generate_test_data()
    BAKSrate = BAKS.autoBAKS(Spikes, Time)
    if BAKSrate is None:
        print("autoBAKS failed to fit array")
    else:
        print("autoBAKS array fit success")

    print("testing autoBAKS on simulated dataframe with multiple units")
    df = generate_sim_df()
    df['BAKSrate'] = BAKS.autoBAKS(df['Spikes'], df['Time'])
    if df['BAKSrate'].isnull().values.any():
        print("autoBAKS failed to fit all units")
    else:
        print("autoBAKS dataframe fit success")

    print("testing autoBAKS on simulated list of numpy arrays for multiple units")
    Spikes = df['Spikes'].tolist()
    Time = df['Time'].tolist()
    BAKSrate = BAKS.autoBAKS(Spikes, Time)
    if BAKSrate is None:
        print("autoBAKS failed list of numpy arrays test")
    else:
        print("autoBAKS list of numpy arrays test success")

    print("testing autoBAKS on simulated 2D numpy array for multiple units")
    Spikes = np.array(df['Spikes'].tolist())
    Time = np.array(df['Time'].tolist())
    BAKSrate = BAKS.autoBAKS(Spikes, Time)
    if BAKSrate is None:
        print("autoBAKS failed 2D numpy array test")
    else:
        print("autoBAKS 2D numpy array test success")






