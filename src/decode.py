import numpy as np

# class cipher(permuted_alphabet):
#     self.permuted_alphabet:

def decode(ciphertext: str, has_breakpoint: bool) -> str:
    plaintext = ciphertext  # Replace with your code
    return plaintext


with open('data/alphabet.csv', 'r') as f:
    letters = f.read()
letters = letters.strip().split(',')
print(letters)
