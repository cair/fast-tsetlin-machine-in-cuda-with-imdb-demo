# Tsetlin Machine with Bitwise Operators Implemented in CUDA

A CUDA implementation of the Tsetlin Machine (https://arxiv.org/abs/1804.01508) using bitwise operations for increased learning- and classification speed. On the IMDB dataset, parallel bit manipulation with CUDA leads to approx. 50 times faster learning compared to the vanilla Cython (https://github.com/cair/TsetlinMachine) and C (https://github.com/cair/TsetlinMachineC) implementations.

## Bit-Based Representation and Manipulation of Patterns

The Tsetlin Machine solves complex pattern recognition problems with propositional formulas, composed by a collective of Tsetlin Automata. In this implementation, we express both inputs, patterns, and outputs as bits, while recognition and learning rely on bit manipulation. Briefly stated, the states of the Tsetlin Automata are jointly represented using multiple sequences of bits (e.g., 8 sequences to represent an 8 bit state index). Sequence 1 contains the first bit of each state index. Sequence 2 contains the second bit, and so on, as exemplified below for 24 Tsetlin Automata:

![Figure 4](https://github.com/olegranmo/blob/blob/master/Bit_Manipulation_3.png)

The benefit of this representation is that the action of each Tsetlin Automaton is readily available from the most significant bit (sequence 8 in the figure). Thus, the output (recognized or not recognized pattern) can be obtained from the input based on fast bitwise operators (NOT, AND, and CMP - comparison). When deployed after training, only the sequence containing the most significant bit is required. The other sequences can be discarded because these bits are only used to keep track of the learning. This provides a further reduction in memory usage.

## IMDB Demo
```bash
python ./produce_dataset.py
make
./IMDBDemoCUDABits

Num SMS: 80

EXPERIMENT 1

##### EPOCH 1 #####

-- CLASS 1 --

PRECISION: 0.876
RECALL: 0.856
F-SCORE: 0.866

-- CLASS 2 --

PRECISION: 0.859
RECALL: 0.878
F-SCORE: 0.869

TRAINING TIME: 28.366195
TESTING TIME: 19.398722

##### EPOCH 2 #####

-- CLASS 1 --

PRECISION: 0.874
RECALL: 0.869
F-SCORE: 0.872

-- CLASS 2 --

PRECISION: 0.870
RECALL: 0.875
F-SCORE: 0.872

TRAINING TIME: 29.422352
TESTING TIME: 19.346641

##### EPOCH 3 #####

-- CLASS 1 --

PRECISION: 0.877
RECALL: 0.877
F-SCORE: 0.877

-- CLASS 2 --

PRECISION: 0.877
RECALL: 0.877
F-SCORE: 0.877

TRAINING TIME: 29.098980
TESTING TIME: 19.351734
```
## Further Work

* Perform a more extensive hyperparameter search (manipulating THRESHOLD, CLAUSES, STATE_BITS, and S in TsetlinMachineConfig.h).
* Convolutional approach for more fine-grained modelling of semantics.
