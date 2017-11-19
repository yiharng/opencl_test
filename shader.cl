#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void adder(__global const float* a, __global const float* b, __global float* result)
{
	int idx = get_global_id(0);
	int i,j;
//	result[idx] = a[idx] + b[idx];
//	result[idx] = 0;
//	for (i = 0; i < 100; i++)
	{
		result[idx] += sin(sqrt(a[idx] * a[idx] + b[idx] * b[idx]) / (b[idx] + 1));
/*
	}
	for (int j = 0; j < 100; j++)
	{
*/
		result[idx] += cos(sqrt(a[idx] * a[idx] + 2*b[idx] * b[idx]) / (a[idx] + 1));
	}
}
