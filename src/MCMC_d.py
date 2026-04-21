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

logLLH_list = []
logLLH_current = compute_logLLH(current_inv_cipher_dict , y, letter_to_int)
acceptance_history = np.zeros(n_steps)
accuracy_list = []

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
    accuracy = np.sum( np.array(list(current_decoded_ciphertext)) == np.array(list(gt_text))) / len(gt_text)
    accuracy_list.append(accuracy)



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

# #### Plotting for Problem 3a
# logLLH_arr = np.asarray(logLLH_list)
# iters = np.arange(n_steps)
# burn_in_end = int(2_050)

# fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
# if burn_in_end > 0:
#     ax.axvspan(
#         0,
#         burn_in_end,
#         facecolor="#f5d5c8",
#         alpha=0.55,
#         zorder=0,
#         label="Burn-in phase",
#     )
# if burn_in_end < n_steps:
#     ax.axvspan(
#         burn_in_end,
#         n_steps,
#         facecolor="#c8dff5",
#         alpha=0.45,
#         zorder=0,
#         label="Near stationarity distribution - sample region",
#     )
# ax.plot(iters, logLLH_arr, color="#1f4e79", linewidth=0.9, zorder=2, label="Log-likelihood trace")
# ax.set_xlabel("Iteration")
# ax.set_ylabel("Log-likelihood of current state")
# ax.set_title("MCMC LLH of accepted state vs. iteration count")
# ax.grid(True, alpha=0.35, zorder=1)
# ax.legend(loc="lower right", framealpha=0.92)

# fig.savefig("mcmc_convergence_plot.png", dpi=300, bbox_inches="tight")
# plt.show()
# ####

# ## Problem 3 b

# acc_rate_list = []

# for t in range(n_steps):
#     if t < T-1:
#         acc_rate_list.append(np.nan)
#     else:
#         slice_acc = acceptance_history[t-T+1 : t+1]
#         acc_rate = np.mean(slice_acc)
#         acc_rate_list.append(acc_rate)

# acc_rate_arr = np.asarray(acc_rate_list, dtype=float)

# pct_fmt = plt.matplotlib.ticker.PercentFormatter(xmax=1)

# # figure 2: full y scale (same layout / colors as Problem 3a)
# fig2, ax2 = plt.subplots(figsize=(9, 4.5), layout="constrained")
# if burn_in_end > 0:
#     ax2.axvspan(
#         0,
#         burn_in_end,
#         facecolor="#f5d5c8",
#         alpha=0.55,
#         zorder=0,
#         label="Burn-in phase",
#     )
# if burn_in_end < n_steps:
#     ax2.axvspan(
#         burn_in_end,
#         n_steps,
#         facecolor="#c8dff5",
#         alpha=0.45,
#         zorder=0,
#         label="Near stationarity distribution - sample region",
#     )
# ax2.plot(
#     iters,
#     acc_rate_arr,
#     color="#1f4e79",
#     linewidth=0.9,
#     zorder=2,
#     label=f"Acceptance rate (window T={T})",
# )
# ax2.set_xlabel("Iteration")
# ax2.set_ylabel("Acceptance rate")
# ax2.set_title(f"MCMC sliding-window acceptance rate (T={T})")
# ax2.set_ylim(0, 0.2)
# ax2.yaxis.set_major_formatter(pct_fmt)
# ax2.grid(True, alpha=0.35, zorder=1)
# ax2.legend(loc="upper right", framealpha=0.92)
# fig2.savefig(PROJECT_ROOT / "3b_acceptance_rate_plot.png", dpi=300, bbox_inches="tight")
# plt.show()

# # figure 3: same style, y zoomed to 0–4%
# fig3, ax3 = plt.subplots(figsize=(9, 4.5), layout="constrained")
# if burn_in_end > 0:
#     ax3.axvspan(
#         0,
#         burn_in_end,
#         facecolor="#f5d5c8",
#         alpha=0.55,
#         zorder=0,
#         label="Burn-in phase",
#     )
# if burn_in_end < n_steps:
#     ax3.axvspan(
#         burn_in_end,
#         n_steps,
#         facecolor="#c8dff5",
#         alpha=0.45,
#         zorder=0,
#         label="Near stationarity distribution - sample region",
#     )
# ax3.plot(
#     iters,
#     acc_rate_arr,
#     color="#1f4e79",
#     linewidth=0.9,
#     zorder=2,
#     label=f"Acceptance rate (window T={T})",
# )
# ax3.set_xlabel("Iteration")
# ax3.set_ylabel("Acceptance rate")
# ax3.set_title(f"MCMC sliding-window acceptance rate (T={T})")
# ax3.set_ylim(0, 0.04)
# ax3.yaxis.set_major_formatter(pct_fmt)
# ax3.grid(True, alpha=0.35, zorder=1)
# ax3.legend(loc="upper right", framealpha=0.92)
# fig3.savefig(PROJECT_ROOT / "3b_acceptance_rate_zoom.png", dpi=300, bbox_inches="tight")
# plt.show()

# ###################

# ##plot for 3c

# accuracy_arr = np.asarray(accuracy_list, dtype=float)

# fig4, ax4 = plt.subplots(figsize=(9, 4.5), layout="constrained")
# if burn_in_end > 0:
#     ax4.axvspan(
#         0,
#         burn_in_end,
#         facecolor="#f5d5c8",
#         alpha=0.55,
#         zorder=0,
#         label="Burn-in phase",
#     )
# if burn_in_end < n_steps:
#     ax4.axvspan(
#         burn_in_end,
#         n_steps,
#         facecolor="#c8dff5",
#         alpha=0.45,
#         zorder=0,
#         label="Near stationarity distribution - sample region",
#     )
# ax4.plot(
#     iters,
#     accuracy_arr,
#     color="#1f4e79",
#     linewidth=0.9,
#     zorder=2,
#     label="Decoding accuracy",
# )
# ax4.axhline(1.0, color="#2a8a3e", linestyle="--", linewidth=0.8, zorder=1, label="Perfect decoding")
# ax4.set_xlabel("Iteration")
# ax4.set_ylabel("Decoding accuracy")
# ax4.set_title("MCMC decoding accuracy vs. iteration count")
# ax4.set_ylim(0, 1.05)
# ax4.yaxis.set_major_formatter(pct_fmt)
# ax4.grid(True, alpha=0.35, zorder=1)
# ax4.legend(loc="lower right", framealpha=0.92)
# fig4.savefig(PROJECT_ROOT / "3c_accuracy_plot.png", dpi=300, bbox_inches="tight")
# plt.show()

# # figure 5: same style, y zoomed to 98–100%
# fig5, ax5 = plt.subplots(figsize=(9, 4.5), layout="constrained")
# if burn_in_end > 0:
#     ax5.axvspan(
#         0,
#         burn_in_end,
#         facecolor="#f5d5c8",
#         alpha=0.55,
#         zorder=0,
#         label="Burn-in phase",
#     )
# if burn_in_end < n_steps:
#     ax5.axvspan(
#         burn_in_end,
#         n_steps,
#         facecolor="#c8dff5",
#         alpha=0.45,
#         zorder=0,
#         label="Near stationarity distribution - sample region",
#     )
# ax5.plot(
#     iters,
#     accuracy_arr,
#     color="#1f4e79",
#     linewidth=0.9,
#     zorder=2,
#     label="Decoding accuracy",
# )
# ax5.axhline(1.0, color="#2a8a3e", linestyle="--", linewidth=0.8, zorder=1, label="Perfect decoding")
# ax5.set_xlabel("Iteration")
# ax5.set_ylabel("Decoding accuracy")
# ax5.set_title("MCMC decoding accuracy vs. iteration count (zoom)")
# ax5.set_ylim(0.92, 1.0)
# ax5.yaxis.set_major_formatter(pct_fmt)
# ax5.grid(True, alpha=0.35, zorder=1)
# ax5.legend(loc="lower right", framealpha=0.92)
# fig5.savefig(PROJECT_ROOT / "3c_accuracy_zoom.png", dpi=300, bbox_inches="tight")
# plt.show()


# #reversing if necessary with
# f_dict = {value: key for key, value in current_inv_cipher_dict.items()}


### Problem 3d

segment_length = 1000
n_steps = 5000
T=500
initial_inv_cipher_dict = {letters[i]: letters[(i+1) % len(letters)] for i in range(len(letters))}


def average_performance(segment_length, n_steps, T, initial_inv_cipher_dict):
    segments_y = []
    segments_gt = []
    for i in range(0, len(y), segment_length):
        if len(y[i:i+segment_length]) == segment_length: # drop the last uneven segment
            segments_y.append(y[i:i+segment_length])
            segments_gt.append(gt_text[i:i+segment_length])




    all_logLLH = []
    all_acc_rate = []
    all_accuracy = []

    for seg_y, seg_gt in zip(segments_y, segments_gt):
        logLLH_list, acceptance_history, accuracy_list = MCMC(n_steps, initial_inv_cipher_dict, seg_y, letter_to_int, seg_gt)

        accuracy_arr = np.asarray(accuracy_list, dtype=float)
        logLLH_arr = np.asarray(logLLH_list, dtype=float)

        # calculate acceptance rate
        acc_rate_list = []

        for t in range(n_steps):
            if t < T-1:
                acc_rate_list.append(np.nan)
            else:
                slice_acc = acceptance_history[t-T+1 : t+1]
                acc_rate = np.mean(slice_acc)
                acc_rate_list.append(acc_rate)

        acc_rate_arr = np.asarray(acc_rate_list, dtype=float)

        # put the arrays into the lists
        all_logLLH.append(logLLH_arr)
        all_acc_rate.append(acc_rate_arr)
        all_accuracy.append(accuracy_arr)

    stacked_logLLH = np.stack(all_logLLH)
    stacked_acc_rate = np.stack(all_acc_rate)
    stacked_accuracy = np.stack(all_accuracy)

    avg_logLLH = np.mean(stacked_logLLH, axis=0)
    avg_acc_rate = np.mean(stacked_acc_rate, axis=0)
    avg_accuracy = np.mean(stacked_accuracy, axis=0)

    return avg_logLLH, avg_acc_rate, avg_accuracy


iters = range(n_steps)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

for segment_length in [50, 100, 250, 500, 1000, 2500 ,5287]:
    avg_logLLH, avg_acc_rate, avg_accuracy = average_performance(segment_length, n_steps, T, initial_inv_cipher_dict)

    # axes[0].plot(iters, avg_logLLH, label=f'Segment length = {segment_length}')
    axes[0].plot(iters, avg_acc_rate, label=f'Segment length = {segment_length}')
    axes[1].plot(iters, avg_accuracy, label=f'Segment length = {segment_length}')

# axes[0].set_title('Average Log-Likelihood')
# axes[0].set_ylabel('Log-Likelihood')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)

axes[0].set_title(f'Average Acceptance Rate (T={T})')
axes[0].set_ylabel('Acceptance Rate')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Average Decoding Accuracy')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(PROJECT_ROOT / "3d_segment_length_plot.png", dpi=300, bbox_inches="tight")
plt.show()

#TODO: Write 3d explanation
