// OpenCL tutorial 1

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "time.h"
#include "math.h"
#include "process.h"
#include "windows.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//#pragma comment(lib,"F:\\ccc\\opencl\\NVIDIA GPU Computing SDK\\OpenCL\\common\\lib\\Win32\\OpenCL.lib")
#pragma comment(lib,"F:\\ccc\\opencl\\NVIDIA GPU Computing SDK\\OpenCL\\common\\lib\\x64\\OpenCL.lib")
//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

//#define SSS

int list1()
{

	int i, j;
	char* value;
	size_t valueSize;
	cl_uint platformCount;
	cl_platform_id* platforms;
	cl_uint deviceCount;
	cl_device_id* devices;
	cl_uint maxComputeUnits;

	// get all platforms
	clGetPlatformIDs(0, NULL, &platformCount);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	for (i = 0; i < platformCount; i++) 
	{

		// get all devices
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

		// for each device print critical attributes
		for (j = 0; j < deviceCount; j++) {

			// print device name
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
			printf("%d. Device: %s\n", j + 1, value);
			free(value);

			// print hardware device version
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
			printf(" %d.%d Hardware version: %s\n", j + 1, 1, value);
			free(value);

			// print software driver version
			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
			printf(" %d.%d Software version: %s\n", j + 1, 2, value);
			free(value);

			// print c version supported by compiler for device
			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
			printf(" %d.%d OpenCL C version: %s\n", j + 1, 3, value);
			free(value);

			// print parallel compute units
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(maxComputeUnits), &maxComputeUnits, NULL);
			printf(" %d.%d Parallel compute units: %d\n", j + 1, 4, maxComputeUnits);

		}

		free(devices);

	}

	free(platforms);
	return 0;

}

int list2()
{

	int i, j;
	char* info;
	size_t infoSize;
	cl_uint platformCount;
	cl_platform_id *platforms;
	const char* attributeNames[5] = { "Name", "Vendor",
		"Version", "Profile", "Extensions" };
	const cl_platform_info attributeTypes[5] = { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR,
		CL_PLATFORM_VERSION, CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };
	const int attributeCount = sizeof(attributeNames) / sizeof(char*);

	// get platform count
	clGetPlatformIDs(5, NULL, &platformCount);

	// get all platforms
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	// for each platform print all attributes
	for (i = 0; i < platformCount; i++) {

		printf("\n %d. Platform \n", i + 1);

		for (j = 0; j < attributeCount; j++) {

			// get platform attribute value size
			clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
			info = (char*)malloc(infoSize);

			// get platform attribute value
			clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);

			printf("  %d.%d %-11s: %s\n", i + 1, j + 1, attributeNames[j], info);
			free(info);

		}

		printf("\n");

	}

	free(platforms);
	return 0;

}

cl_program load_program(cl_context context, const char* filename)
{
	FILE *f;
	char *data;
	int len;

	f = fopen(filename, "rb");
	fseek(f, 0, SEEK_END);
	len = ftell(f);
	fseek(f, 0, SEEK_SET);
	data = (char*)malloc(len + 8);
	fread(data, 1, len, f);
	data[len] = 0;
	fclose(f);

	//	printf("data=%s\nlen=%d\n", &data[0],len);

		// create and build program 
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&data, 0, 0);

	free(data);

	if (program == 0) {
		return 0;
	}

	if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
		return 0;
	}

	return program;
}

void runcpu(float *a, float *b, float *res, int DATA_SIZE)
{
	for (int i = 0; i < DATA_SIZE; i++)
	{
//		res[i] = 0;
//		for (int j = 0; j < 100; j++)
		{
			res[i] += sinf(sqrtf(a[i] * a[i] + b[i] * b[i]) / (b[i] + 1));
#ifdef SSS
		}
		for (int j = 0; j < 100; j++)
		{
#endif
			res[i] += cosf(sqrtf(a[i] * a[i] + 2 * b[i] * b[i]) / (a[i] + 1));
		}
	}


}

int thrn = 0;

//#define THRRUN

#ifdef THRRUN
void thrrun(float *a, float *b, float *res, int i)
{
	res[i] += sinf(sqrtf(a[i] * a[i] + b[i] * b[i]) / (b[i] + 1));
	res[i] += cosf(sqrtf(a[i] * a[i] + 2 * b[i] * b[i]) / (a[i] + 1));
}
#endif
void runthread(void *k)//double *a, double *b, double *res, int size)
{
	float *a = ((float**)k)[0];
	float *b = ((float**)k)[1];
	float *res = ((float**)k)[2];
	int size = ((float**)k)[3][0];

	for (int i = 0; i < size; i++)
	{
#ifdef THRRUN
		thrrun(a,b,res,i);
#else
//		res[i] = 0;
//		for (int j = 0; j < 100; j++)
		{
			res[i] += sinf(sqrtf(a[i] * a[i] + b[i] * b[i]) / (b[i] + 1));
#ifdef SSS
		}
		for (int j = 0; j < 100; j++)
		{
#endif
			res[i] += cosf(sqrtf(a[i] * a[i] + 2 * b[i] * b[i]) / (a[i] + 1));
		}
#endif
	}
	thrn++;
}

int main()
{
	cl_int err;
	cl_uint num;
	int i;

	printf("===========list1\n");
	list1();
	printf("===========list2\n");
	list2();
	printf("===========main\n");

	err = clGetPlatformIDs(0, 0, &num);
	if (err != CL_SUCCESS) {
		std::cerr << "Unable to get platforms\n";
		getchar();
		return 0;
	}

	cl_platform_id platforms[256];
	err = clGetPlatformIDs(256, platforms, &num);

	printf("platforms=%d\n", num);
	printf("size float=%d\n", sizeof(float));

	if (err != CL_SUCCESS) {
		std::cerr << "Unable to get platform ID\n";
		getchar();
		return 0;
	}

	char *info;
	size_t infoSize;
	for (i = 0; i < num; i++)
	{
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &infoSize);
		info = (char*)malloc(infoSize);

		// get platform attribute value
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, infoSize, info, NULL);

		printf("%d - %s\n", i, info);
		free(info);
	}

	cl_device_id cdDevice;
	//int ciErr1 = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL);
	int ciErr1 = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	//int ciErr1 = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	if (ciErr1 != CL_SUCCESS)
	{
		printf("error get device id\n");
		getchar();
		exit(0);
	}
	char devname[1024];
	clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, 1024, devname, 0);
	printf("devname=%s\n", devname);

	//getchar();

	/*
	cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
	cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
	if (context == 0) {
		std::cerr << "Can't create OpenCL context\n";
		getchar();
		return 0;
	}
	//*/

	cl_context context = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);

	size_t nDeviceBytes;
	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &nDeviceBytes);
	int ciDeviceCount = (cl_uint)nDeviceBytes / sizeof(cl_device_id);
	printf("device byte=%d   device count=%d\n", nDeviceBytes, ciDeviceCount);

	cl_command_queue queue = clCreateCommandQueue(context, cdDevice, 0, 0);
	if (queue == 0) {
		std::cerr << "Can't create command queue\n";
		clReleaseContext(context);
		getchar();
		return 0;
	}

	const int DATA_SIZE = 1048576;

	float *a, *b, *res, *res1, *res2;

	a = (float*)calloc(1,sizeof(float)*DATA_SIZE);
	b = (float*)calloc(1,sizeof(float)*DATA_SIZE);
	res = (float*)calloc(1,sizeof(float)*DATA_SIZE);
	res1 = (float*)calloc(1,sizeof(float)*DATA_SIZE);
	res2 = (float*)calloc(1,sizeof(float)*DATA_SIZE);

	printf("RAND_MAX=%d\n", RAND_MAX);
	for (int i = 0; i < DATA_SIZE; i++)
	{
		a[i] = rand() / (float)RAND_MAX;
		b[i] = rand() / (float)RAND_MAX;
		res[i] = 0;
	}
	for (int i = 0; i < 10; i++)
	{
		a[i] = i;
		b[i] = i;
	}

	for (int i = 0; i < 20; i++)
	{
		printf("%f\t", a[i]);
	}
	printf("\n");

	clock_t t1;
	clock_t time1;
	clock_t time2;

	t1 = clock();
	for (int v = 0; v < 100; v++)
	{
		runcpu(&a[0], &b[0], &res1[0], DATA_SIZE);
	}
	time1 = clock() - t1;
	printf("time1:%d\n", time1);

	///////////////////////////////
	float *k[32][10];
	float ks[32][2];
	int nk = 4;
	int ds4 = DATA_SIZE / nk;
	int ki;

	for (ki = 0; ki < nk; ki++)
	{
		k[ki][0] = a + ds4*ki;
		k[ki][1] = b + ds4*ki;
		k[ki][2] = res2 + ds4*ki;
		k[ki][3] = (float *)(ks[ki]);
		ks[ki][0] = ds4;
	}

	ks[ki - 1][0] = DATA_SIZE - ds4*(ki - 1);

	t1 = clock();
	for (int v = 0; v < 100; v++)
	{
		thrn = 0;
		for (ki = 0; ki < nk; ki++)
		{
			_beginthread(runthread, 0, k[ki]);
		}
		while (thrn < nk)
		{
			Sleep(1);
		}
	}
	time2 = clock() - t1;
	printf("time2:%d\n", time2);
	printf("speed:%lf\n", (double)time1 / time2);

	{
		bool correct = true;
		for (int i = DATA_SIZE - 1; i >= 0; i--) {
			if (res2[i] != res1[i])
			{
				correct = false;
				printf("i=%d res2=%f   res1=%f\n", i, res2[i], res1[i]);
				if (!i) break;
				printf("i=%d res2=%f   res1=%f\n", i, res2[i - 1], res1[i - 1]);
				printf("i=%d res2=%f   res1=%f\n", i, res2[i - 2], res1[i - 2]);
				break;
			}
		}

		if (correct) {
			std::cout << "Data is correct\n";
		}
		else {
			std::cout << "Data is incorrect\n";
		}
	}
	
	printf("cpu end....\n");
	getchar();


	t1 = clock();
	cl_mem cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &a[0], NULL);
	cl_mem cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &b[0], NULL);
//	cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * DATA_SIZE, NULL, NULL);
	cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &res[0], NULL);
	if (cl_a == 0 || cl_b == 0 || cl_res == 0)
	{
		std::cerr << "Can't create OpenCL buffer\n";
		clReleaseMemObject(cl_a);
		clReleaseMemObject(cl_b);
		clReleaseMemObject(cl_res);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		getchar();
		return 0;
	}

	cl_program program = load_program(context, "shader.cl");
	if (program == 0) {
		std::cerr << "Can't load or build program\n";
		clReleaseMemObject(cl_a);
		clReleaseMemObject(cl_b);
		clReleaseMemObject(cl_res);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		getchar();
		return 0;
	}

	cl_kernel adder = clCreateKernel(program, "adder", 0);
	if (adder == 0)
	{
		std::cerr << "Can't load kernel\n";
		clReleaseProgram(program);
		clReleaseMemObject(cl_a);
		clReleaseMemObject(cl_b);
		clReleaseMemObject(cl_res);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		getchar();
		return 0;
	}

	clSetKernelArg(adder, 0, sizeof(cl_mem), &cl_a);
	clSetKernelArg(adder, 1, sizeof(cl_mem), &cl_b);
	clSetKernelArg(adder, 2, sizeof(cl_mem), &cl_res);


	size_t work_size = DATA_SIZE;
	size_t localWorkSize[] = { 256 };

	printf("time000:%d\n", clock() - t1);
	t1 = clock();
	for (int v = 0; v < 100; v++)
	{
		err = clEnqueueNDRangeKernel(queue, adder, 1, 0, &work_size, localWorkSize, 0, 0, 0);
	}
	printf("time:%d\n", clock() - t1);
	t1 = clock();
	if (err == CL_SUCCESS) {
		err = clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(float) * DATA_SIZE, &res[0], 0, 0, 0);
	}
	printf("time111:%d\n", clock() - t1);

	for (int i = 0; i < 20; i++)
	{
		printf("%f\t", res[i]);
	}

	printf("\n");
//	Sleep(2000);

	if (err == CL_SUCCESS)
	{
		bool correct = true;
		for (int i = DATA_SIZE - 1; i >= 0; i--) {
			if (res1[i] != res[i])
			{
				correct = false;
				printf("i=%d res1=%f   res=%f\n", i, res1[i], res[i]);
				printf("i=%d res1=%f   res=%f\n", i, res1[i - 1], res[i - 1]);
				printf("i=%d res1=%f   res=%f\n", i, res1[i - 2], res[i - 2]);
				break;
			}
		}

		if (correct) {
			std::cout << "Data is correct\n";
		}
		else {
			std::cout << "Data is incorrect\n";
		}
	}
	else {
		std::cerr << "Can't run kernel or read back data\n";
	}

	clReleaseKernel(adder);
	clReleaseProgram(program);
	clReleaseMemObject(cl_a);
	clReleaseMemObject(cl_b);
	clReleaseMemObject(cl_res);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	getchar();
	return 0;
}

