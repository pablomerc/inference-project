import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

random.seed(43)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

with open(DATA_DIR / "alphabet.csv", "r") as f:
    letters = f.read()
letters = letters.strip().split(",")

with open(DATA_DIR / "sample" / "ciphertext.txt", "r") as f:
    y = f.read().rstrip("\n")

with open(DATA_DIR / "sample" / "plaintext.txt", "r") as f:
    gt_text = f.read().rstrip("\n")

print(len(y))

# print(y)

# P vector
letter_probabilities = np.loadtxt(DATA_DIR / "letter_probabilities.csv", delimiter=",")
# print(type(letter_probabilities), letter_probabilities.shape)

# M matrix
letter_transition_matrix = np.loadtxt(DATA_DIR / "letter_transition_matrix.csv", delimiter=",")
# print(type(letter_transition_matrix), letter_transition_matrix.shape)


# print(letters)

# First cipher is arbitrary lets just map to the next letter
# Note that it's arbitrary to define f^-1 or f and its easier to work in f^-1 space
current_inv_cipher_dict = {letters[i]: letters[(i+1) % len(letters)] for i in range(len(letters))}

# Define translation from a letter to an index of the P vector and M matrix
letter_to_int = {letters[i]: i for i in range(len(letters))}

print(current_inv_cipher_dict)

def cipher_proposal(current_cipher):
    """Take the current cipher as input, produce a cipher with only 2 diff symbols as output"""
    next_cipher = current_cipher.copy()

    #note that random.saple samples without replacement :)
    #so we are guaranteed different keys get sampled here
    key1, key2 = random.sample(list(next_cipher.keys()), 2)

    #Swap two elements of next_cipher - we generate a cipher with only 2 diff symbols
    next_cipher[key1], next_cipher[key2] = next_cipher[key2], next_cipher[key1]
    #finding f1,f2 that differ on two elements is the same as finding f1^-1 and f2^-1 that differ on two elements

    return next_cipher

#Take the log once
log_M = np.log(letter_transition_matrix + 1e-10)
log_P = np.log(letter_probabilities + 1e-10)
def compute_logLLH(f_inv_dict, y, letter_to_int):
    """calculate log likelihood of the ciphertext y under the given f^-1 """
    decoded_ints = [letter_to_int[f_inv_dict[char]] for char in y]

    # will use numpy arrays to make things faster
    ints_array = np.array(decoded_ints)

    transitions = log_M[ints_array[1:],ints_array[:-1]] # this gets the k and k-1 letters each time

    return log_P[ints_array[0]] + np.sum(transitions)

# def acceptance_factor(f_inv, f_inv_next, y, letter_to_int):
#     """take the current cipher (or inverse cipher), the proposed "next" one, and the ciphertext y
#     # and calculate a(f->f')"""

#     logLLH_current = compute_logLLH(f_inv, y, letter_to_int)
#     logLLH_next = compute_logLLH(f_inv_next, y, letter_to_int)

#     log_diff = logLLH_next - logLLH_current
#     if log_diff >= 0:
#         return 1.0

#     else:
#         return np.exp(log_diff)

def bernoulli(p):
    """return true if a random value between 0 and 1 is within [0,p), which happens 100p% of the time"""
    return random.random() < p


def decode_ciphertext(f_inv,y):
    return ''.join([f_inv[char] for char in y])


n_steps = 5_000
T = 500


def MCMC(n_steps, initial_inv_cipher_dict,y,letter_to_int, gt):
    logLLH_list = []
    logLLH_current = compute_logLLH(initial_inv_cipher_dict , y, letter_to_int)
    acceptance_history = np.zeros(n_steps)
    accuracy_list = []
    current_inv_cipher_dict = initial_inv_cipher_dict.copy()

    for i in range(n_steps):
        next_inv_cipher_dict = cipher_proposal(current_inv_cipher_dict)
        # print('Next cipher', next_inv_cipher_dict)

        logLLH_next = compute_logLLH(next_inv_cipher_dict , y, letter_to_int)
        log_diff = logLLH_next - logLLH_current

        a = 1.0 if log_diff >= 0 else np.exp(log_diff)

        if bernoulli(a):

            current_inv_cipher_dict = next_inv_cipher_dict
            logLLH_current = logLLH_next

            #for 3b
            acceptance_history[i] = 1

        logLLH_list.append(logLLH_current)
        #for 3c
        current_decoded_ciphertext = decode_ciphertext(current_inv_cipher_dict, y)
        accuracy = np.sum( np.array(list(current_decoded_ciphertext)) == np.array(list(gt))) / len(gt)
        accuracy_list.append(accuracy)

    return logLLH_list, acceptance_history, accuracy_list


# Problem 3e

logLLH_list, acceptance_history, accuracy_list = MCMC(n_steps, current_inv_cipher_dict,y,letter_to_int, gt_text)
logLLH = np.asarray(logLLH_list)

# convert nats -> bits/letter to match Shannon's units
entropy = -logLLH / (np.log(2) * len(y))

uniform_entropy = np.log2(len(letters))

iters = np.arange(n_steps)
burn_in_end = 2_050

fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
if burn_in_end > 0:
    ax.axvspan(0, burn_in_end, facecolor="#f5d5c8", alpha=0.55, zorder=0, label="Burn-in phase")
if burn_in_end < n_steps:
    ax.axvspan(burn_in_end, n_steps, facecolor="#c8dff5", alpha=0.45, zorder=0,
               label="Near stationarity distribution - sample region")

ax.plot(iters, entropy, color="#1f4e79", linewidth=0.9, zorder=3,
        label="Per-symbol entropy of current state")

ax.axhline(uniform_entropy, color="#8e44ad", linestyle="--", linewidth=1.2, zorder=2,
           label=f"Uniform over {len(letters)} symbols ({uniform_entropy:.2f} bits)")
ax.axhline(3.32, color="#2a8a3e", linestyle="--", linewidth=1.2, zorder=2,
           label=r"Shannon $F_2$ = 3.32 bits/letter (27-letter)")
ax.axhline(2.3, color="#c0392b", linestyle="--", linewidth=1.2, zorder=2,
           label=r"Shannon $F_8 \approx$ 2.3 bits/letter")

ax.set_xlabel("Iteration")
ax.set_ylabel("Entropy rate (bits / symbol)")
ax.set_title("MCMC per-symbol entropy of decoded text vs. iteration")
ax.grid(True, alpha=0.35, zorder=1)
ax.legend(loc="upper right", framealpha=0.92, fontsize=9)

fig.savefig(PROJECT_ROOT / "3e_entropy_plot.png", dpi=300, bbox_inches="tight")
plt.show()
