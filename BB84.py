# %%
from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, pi
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import fake_provider
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

mode = "custom_noise" # "custom_noise", "fake", "real"
key_length = 1000 #must be multiple of 1
min_key_length = 20

eavesdropping = False
depolarizing_noise = 0.001
p0g1 = p1g0 = read_out_error = 0.0

# %%

# 0 for +, 1 for x
def exchange(mode, key_length, eavesdropping, depolarizing_noise, read_out_error):
    p1g0 = p0g1 = read_out_error
    alice_nouce  = np.random.randint(0,2,key_length)
    bob_nouce = np.random.randint(0,2,key_length)
    #alice_nouce = bob_nouce = np.random.randint(0,2,key_length)
    
    alice_key = np.random.randint(0,2,key_length)
    bob_key = np.zeros(key_length, dtype=int)
    eve_key = np.zeros(key_length, dtype=int)


    # %%
    # i_op = qi.Operator([[1,0],[0,1]])

    circuits = []
    for i in range(key_length):
        all_qc = QuantumCircuit(1,2)   
        alice_qc = QuantumCircuit(1,0)
        eve_qc = QuantumCircuit(1,2)
        bob_qc = QuantumCircuit(1,1)
        
        # Alice's circuit
        if alice_key[i] == 1:
            alice_qc.x(0)
        if alice_nouce[i] == 1:
            alice_qc.h(0)
        alice_qc.id(0)
        ##### Interception by Eve #####  
        # Eve's circuit
        if eavesdropping:
            eve_qc.u(pi/4,0,0,0).inverse()
            eve_qc.measure(0,1)
            eve_qc.u(pi/4,0,0,0)

        ##### Transporting Qbits to Bob #####
        # Bob's circuit  
        if bob_nouce[i] == 1:
            bob_qc.h(0)
        bob_qc.measure(0,0)
        
        all_qc.compose(alice_qc, inplace=True)
        all_qc.compose(eve_qc, inplace=True)
        all_qc.compose(bob_qc, inplace=True)
        circuits.append(all_qc)

    # %%
    if mode == "custom_noise":
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            depolarizing_error(depolarizing_noise, 1),
            ['id'])
        noise_model.add_all_qubit_readout_error(ReadoutError(
            [[1-p1g0, p1g0], [p0g1, 1-p0g1]]))
        basis_gates = noise_model.basis_gates
        
        backend = AerSimulator(noise_model=noise_model, basis_gates=basis_gates)
        results = backend.run(circuits, shots=1, memory=True).result()
        results = [results.get_memory(circuits[i])[0][::-1] for i in range(key_length)]
        
    # %%
    if mode == "fake":
        
        # Method 1(Recommended): Old snapshot within the module
        # Available model list: https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake_provider
        # Do not choose too large model, or your computer will burn.  
        
        #recommended: Armonk(1bit), Bogota(5bit), Casablanca(7bit), Ourense(5bit), Tokyo(20bit)
        backend = fake_provider.FakeBogotaV2()
        
        # Method 2: Snapshot of current 
        # service = QiskitRuntimeService()
        
        # #print(service.backends()) #print available backends
        
        # backend = service.get_backend('ibm_osaka')
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        circuits = [pm.run(all_qc) for all_qc in circuits] #create isa circuits
        # fig = circuits[0].draw(output='mpl' , interactive=True)
        # plt.show()    

        sampler = Sampler(backend=backend)
        job = sampler.run(circuits, shots=1)
        results = job.result()
        results = [result.data["c"].get_bitstrings()[0][::-1] for result in results]
        

    # %%
    if mode == "real":
        # If you did not previously save your credentials, use the following line instead:
        # service = QiskitRuntimeService(channel="ibm_quantum", token="plz put your token here")
        service = QiskitRuntimeService(channel='ibm_quantum')    
        
        backend = service.least_busy(operational=True, simulator=False)
        print(backend)
        
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        circuits = [pm.run(all_qc) for all_qc in circuits] #create isa circuits
        # fig = circuits[0].draw(output='mpl' , interactive=True)
        # plt.show()    

        sampler = Sampler(backend=backend)
        job = sampler.run(circuits, shots=1)
        results = job.result()
        results = [result.data["c"].get_bitstrings()[0][::-1] for result in results]

    # %%


    # print(result)

    #I should reverese it to get the correct order, IDK why it's reversed
    for i in range(key_length):
        bob_key[i] = int(results[i][0])
        eve_key[i] = int(results[i][1])

    # %%

    real_key_len = 0
    bob_match_len = 0
    eve_match_len = 0
    for i in range(key_length):
        if alice_nouce[i] == bob_nouce[i]:
            # print(f"key: {alice_key[i]}, A: {alice_nouce[i]}, E: {eve_key[i]}, B: {bob_key[i]}")
            real_key_len += 1
            if alice_key[i] == bob_key[i]:
                bob_match_len += 1
            if alice_key[i] == eve_key[i]:
                eve_match_len += 1
    
    return real_key_len, bob_match_len, eve_match_len
# print(f"Real key length: {real_key_len}, eve match rate: {eve_match_len/real_key_len}, bob match rate: {bob_match_len/real_key_len}")
