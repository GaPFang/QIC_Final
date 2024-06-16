from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CSwapGate
import matplotlib.pyplot as plt
import numpy as np
from qiskit_aer import AerSimulator
import random
from qiskit_ibm_runtime import fake_provider
from qiskit_ibm_runtime import SamplerV2 as Sampler

def int_to_binArr(n, length):
    return [int(x) for x in list(bin(n)[2:].zfill(length))]

def main():
    noise = False
    
    simulator = AerSimulator(method='statevector')
    # prepare 2M pubkeys for one classical bit
    # k: private key, |f_k>: public key
    M = 8 # number of k
    n = 2 # qubits of f_k
    L = pow(2, n) # length of k
    oracle_dict = np.concatenate((np.zeros(1), np.random.permutation(range(1, pow(2, L))))).astype(int)
    k_zero = np.array([random.getrandbits(L) for _ in range(M)])
    k_one = np.array([random.getrandbits(L) for _ in range(M)])
    for i in range(M):
        while k_zero[i] == 0:
            k_zero[i] = random.getrandbits(L)
        while k_one[i] == 0:
            k_one[i] = random.getrandbits(L)
    f_k_zero = oracle_dict[k_zero]
    f_k_one = oracle_dict[k_one]
    print(f'f_k_zero: {f_k_zero}')
    print(f'f_k_one: {f_k_one}')
    f_k_zero = np.array([int_to_binArr(f_k_zero[i], L) for i in range(M)])
    f_k_one = np.array([int_to_binArr(f_k_one[i], L) for i in range(M)])
    f_k_vector_zero = np.array([f_k_zero[i] / pow(np.sum(f_k_zero[i]), 0.5) for i in range(M)])
    f_k_vector_one = np.array([f_k_one[i] / pow(np.sum(f_k_one[i]), 0.5) for i in range(M)])
    pubkey_zero = [QuantumCircuit(n, name='pubkey_zero') for _ in range(M)]
    pubkey_one = [QuantumCircuit(n, name='pubkey_one') for _ in range(M)]
    for i in range(M):
        pubkey_zero[i].initialize(f_k_vector_zero[i], range(n))
        pubkey_one[i].initialize(f_k_vector_one[i], range(n))

    # Alice's classical bit
    b = random.randint(0, 1)
    print(f'b: {b}')
    msg = [b]
    for i in range(M):
        if b == 0:
            msg.append(k_zero[i])
        else:
            msg.append(k_one[i])

    # Bob's verification
    msg_k = msg[1:]
    msg_f_k = oracle_dict[msg_k]
    print(f'msg_f_k: {msg_f_k}')
    msg_f_k = np.array([int_to_binArr(msg_f_k[i], L) for i in range(M)])
    msg_f_k_vector = np.array([msg_f_k[i] / pow(np.sum(msg_f_k[i]), 0.5) for i in range(M)])
    swap_test = [QuantumCircuit(2 * n + 1, 1, name='swap_test') for _ in range(M)]
    print()
    for i in range(M):
        swap_test[i].reset(0)
        swap_test[i].initialize(msg_f_k_vector[i], range(1, n + 1))
        if b == 0:
            swap_test[i].compose(pubkey_zero[i], range(n + 1, 2 * n + 1), inplace=True)
        else:
            swap_test[i].compose(pubkey_one[i], range(n + 1, 2 * n + 1), inplace=True)
        swap_test[i].h(0)
        for j in range(n):
            swap_test[i].append(CSwapGate(), [0, j + 1, n + j + 1])
        swap_test[i].h(0)
        swap_test[i].measure(0, 0)
        print(swap_test[i])
    # measurement
    if not noise:
        shot = 1024
        result = simulator.run(transpile(swap_test, simulator), shots=shot).result()
        counts = result.get_counts()
        print(counts)
        plt.figure(figsize=(20, 10))
        for i in range(M):
            plt.subplot(2, 4, i + 1)
            plt.bar(counts[i].keys(), counts[i].values())
            plt.title(f'Bob\'s verification {i}')
        plt.show()
    else:
        shot = 1024
        backend = fake_provider.FakeBogotaV2()
        sampler = Sampler(backend=backend)
        job = sampler.run(transpile(swap_test, backend), shots=shot)
        results = job.result()
        count = [results[i].data["c"].get_counts() for i in range(M)]
        print(count)
        plt.figure(figsize=(20, 10))
        for i in range(M):
            plt.subplot(2, 4, i + 1)
            plt.bar(count[i].keys(), count[i].values())
            plt.title(f'Bob\'s verification {i}')
        plt.show()

if __name__ == '__main__':
    main()