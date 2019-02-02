IMDBDemoCUDABits: TsetlinMachineIMDB.cu MultiClassTsetlinMachine.cuh TsetlinMachineKernels.cu TsetlinMachineKernels.cuh TsetlinMachine.cuh TsetlinMachine.cu TsetlinMachineConfig.cuh
	nvcc -o IMDBDemoCUDABits TsetlinMachine.cu TsetlinMachineIMDB.cu TsetlinMachineKernels.cu

clean:
	rm *.o IMDBDemoCUDABits
