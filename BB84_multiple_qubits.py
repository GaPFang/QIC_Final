# %%
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
import numpy as np
from qiskit.primitives.sampler import Sampler
from qiskit_aer import AerSimulator
import random

key_length = 10
min_key_length = 20
eavesdropping = True

# %%
sampler = Sampler()

simulator = AerSimulator(method='statevector')  ##change to density_matrix for further research
all_qc = QuantumCircuit(key_length,2 * key_length)

# 0 for +, 1 for x
alice_nonce = np.random.randint(0,2,key_length)
bob_nonce = np.random.randint(0,2,key_length)
# alice_nonce = bob_nonce = np.random.randint(0,2,key_length)
alice_key = np.random.randint(0,2,key_length)
#alice_key = np.zeros(key_length, dtype=int)
bob_key = np.zeros(key_length, dtype=int)
eve_key = np.zeros(key_length, dtype=int)

#print(alice_nonce, bob_nonce, alice_key)

# %%
# Alice's circuit
# for i in range(key_length):
alice_qc = QuantumCircuit(key_length,0)
bob_qc = QuantumCircuit(key_length,2*key_length)
eve_qc = QuantumCircuit(key_length,2*key_length)
for i in range(key_length):

    if alice_key[i] == 1:
        alice_qc.x(i)
    else :
        alice_qc.id(i)
    if alice_nonce[i] == 1:
        alice_qc.h(i)
    else:
        alice_qc.id(i)

    if eavesdropping:
        eve_qc.u(np.pi/4,0,0,i).inverse()

for i in range(key_length):
    if eavesdropping:
        eve_qc.measure(i,i)
        for j in range(key_length - i - 1):
            eve_qc.id(i)
        eve_qc.u(np.pi/4,0,0,i)

    ##### Transporting Qbits to Bob #####

    # Bob's circuit
    if bob_nonce[i] == 1:
        bob_qc.h(i)
    else:
        bob_qc.id(i)
    bob_qc.measure(i,key_length + i)

all_qc.reset([i for i in range(key_length)])
all_qc.compose(alice_qc, inplace=True)
all_qc.compose(eve_qc, inplace=True)
all_qc.compose(bob_qc, inplace=True)
print(all_qc)
    #circuit = transpile(all_qc, simulator)
result = simulator.run(all_qc, shots=1, memory=True).result().get_memory()[0]
for i in range(key_length):
    bob_key[i] = int(result[i])
    eve_key[i] = int(result[key_length + i])
    
# fig = all_qc.draw(output='mpl' , interactive=True)
# plt.show()


# %%  
# for i in range(key_length):
#     print(f"A's key: {alice_nonce[i]}, B's key: {bob_nonce[i]}, A's gate: {alice_gate[i]}, B's gate: {bob_gate[i]}")
print()
real_key_len = 0
for i in range(key_length):
    if alice_nonce[i] == bob_nonce[i]:
        real_key_len += 1
        print(f"key: {alice_nonce[i]}, A: {alice_key[i]}, B: {bob_key[i]}")
print(f"Real key length: {real_key_len}")