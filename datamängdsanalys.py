import pandas as pd
import numpy as np
from scipy.stats import chisquare, entropy, binomtest

# Logglista för att samla in loggposter
log_entries = []

# Initialiserar dataset1 och dataset2 till None
dataset1 = None
dataset2 = None

# Runs-test: Funktion för att räkna antalet "runs" i en sekvens
def runs_test(data):
    runs = 0
    prev = data[0]
    for val in data[1:]:
        if val != prev:
            runs += 1
        prev = val
    return runs

# Monobit-test: Funktion för att utföra monobit-test
def monobit_test(sequence):
    n_zeros = sequence.tolist().count(0)
    n_ones = sequence.tolist().count(1)
    n = n_zeros + n_ones
    return binomtest(n_zeros, n=n)

# Spectral Test: Funktion för att utföra spektral test med Fouriertransform
def spectral_test(sequence):
    fourier_transform = np.fft.fft(sequence)
    magnitudes = np.abs(fourier_transform)
    return np.mean(magnitudes)

# icke-överlappande Template Matching Test
def non_overlapping_template_matching_test(sequence, template):
    matches = 0
    for i in range(len(sequence) - len(template) + 1):
        if (sequence[i:i+len(template)] == template).all():
            matches += 1
    expected_matches = (len(sequence) - len(template) + 1) * (1 / (2 ** len(template)))
    return chisquare([matches, len(sequence) - matches], [expected_matches, len(sequence) - expected_matches])

# Längsta sekvens av likadana värden
def longest_run_test(sequence):
    max_run = 0
    current_run = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
    return max_run

# Läs in dataseten från CSV-filer
try:
    dataset1 = pd.read_csv('python_PRNG_dataset1.csv', usecols=['Value']).dropna()['Value'].astype(int).values
    dataset2 = pd.read_csv('TRNG_dataset.csv', usecols=['Value']).dropna()['Value'].astype(int).values
    log_entries.append("=== Dataset 1 ===")
    log_entries.append(f'Successfully loaded {len(dataset1)} entries from dataset1.')
    log_entries.append("=== Dataset 2 ===")
    log_entries.append(f'Successfully loaded {len(dataset2)} entries from dataset2.')
except Exception as e:
    log_entries.append(f'Failed to load datasets: {e}')

# Fortsätt analysen om båda datamängderna har laddats framgångsrikt
if dataset1 is not None and dataset2 is not None:
    # Chi-kvadrattest
    log_entries.append("=== Chi-square Test ===")
    freq1 = [(dataset1 == 0).sum(), (dataset1 == 1).sum()]
    freq2 = [(dataset2 == 0).sum(), (dataset2 == 1).sum()]
    chi2, p_chi2 = chisquare(f_obs=freq1, f_exp=freq2)
    log_entries.append(f'Chi-square value: {chi2}')
    log_entries.append(f'p-value (Chi-square): {p_chi2}')

    # Entropitest
    log_entries.append("=== Entropy Test ===")
    entropy1 = entropy(freq1)
    entropy2 = entropy(freq2)
    log_entries.append(f'Entropy for dataset1: {entropy1}')
    log_entries.append(f'Entropy for dataset2: {entropy2}')

    # Runs Test
    log_entries.append("=== Runs Test ===")
    n_runs1 = runs_test(dataset1)
    n_runs2 = runs_test(dataset2)
    log_entries.append(f'Number of runs in dataset1: {n_runs1}')
    log_entries.append(f'Number of runs in dataset2: {n_runs2}')

    # Monobit Test
    log_entries.append("=== Monobit Test ===")
    p_value_mono1 = monobit_test(dataset1)
    p_value_mono2 = monobit_test(dataset2)
    log_entries.append(f'Monobit Test p-value for dataset1: {p_value_mono1}')
    log_entries.append(f'Monobit Test p-value for dataset2: {p_value_mono2}')

    # Spectral Test
    log_entries.append("=== Spectral Test ===")
    spectral_value1 = spectral_test(dataset1)
    spectral_value2 = spectral_test(dataset2)
    log_entries.append(f'Spectral Test mean magnitude for dataset1: {spectral_value1}')
    log_entries.append(f'Spectral Test mean magnitude for dataset2: {spectral_value2}')

    # Non-overlapping Template Matching Test
    log_entries.append("=== Non-overlapping Template Matching Test ===")
    template = np.array([1, 0, 1])
    chi_val1, p_val1 = non_overlapping_template_matching_test(dataset1, template)
    chi_val2, p_val2 = non_overlapping_template_matching_test(dataset2, template)
    log_entries.append(f'Chi value for dataset1: {chi_val1}, p-value: {p_val1}')
    log_entries.append(f'Chi value for dataset2: {chi_val2}, p-value: {p_val2}')

    # Longest-Run Test
    log_entries.append("=== Longest-Run Test ===")
    max_run1 = longest_run_test(dataset1)
    max_run2 = longest_run_test(dataset2)
    log_entries.append(f'Longest run in dataset1: {max_run1}')
    log_entries.append(f'Longest run in dataset2: {max_run2}')

    # Skriver loggen till en fil
    with open('test_log.txt', 'w') as f:
        for entry in log_entries:
            f.write(f"{entry}\n")

    # Skriver ut loggen till terminalen
    print("\n".join(log_entries))


else:
    print("Unable to proceed with analysis. Datasets could not be loaded.")
    print("\n".join(log_entries))
