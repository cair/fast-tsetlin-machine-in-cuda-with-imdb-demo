/*
Copyright (c) 2018 Ole-Christoffer Granmo
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
This code implements the Tsetlin Machine from paper arXiv:1804.01508
https://arxiv.org/abs/1804.01508
*/

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state);


__global__ void type_i_feedback(curandState *state, int *ta_state, int *clause_feedback, int *clause_output, int *Xi, float s);


__global__ void type_ii_feedback(int *ta_state, int *clause_feedback, int *clause_output, int *Xi);


/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
__global__ void sum_up_class_votes(int *clause_output, int *sum);


/* Sum up the votes for each class (this is the multiclass version of the Tsetlin Machine) */
__global__ void generate_clause_feedback(curandState *state, int *clause_feedback, int *class_sum, int target);

__global__ void initialize_clause_output(int *clause_output);

__global__ void calculate_clause_output(int *ta_state, int *clause_output, int *Xi);


__global__ void initialize_clause_output_predict(int *clause_output, int *all_exclude);


__global__ void calculate_clause_output_predict(int *ta_state, int *clause_output, int *all_exclude, int *Xi);


__global__ void update_with_all_exclude(int *clause_output, int *all_exclude);
