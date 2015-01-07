#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include<time.h>

#include <CL/cl.h>

#define DATA_SIZE 256
#define INF 99999
using namespace std;
ofstream outfile,seqfile;

clock_t start, endy;

const char *ProgramSource =
"__kernel void parallel_fw(__global uint * pathDistanceBuffer, __global uint * pathBuffer, const unsigned int numNodes, const unsigned int pass) \n"\
"{ \n"\
    "int xValue = get_global_id(0); \n"\
    "int yValue = get_global_id(1); \n"\
    "int k = pass; \n"\
    "int oldWeight = pathDistanceBuffer[yValue * numNodes + xValue]; \n"\
    "int tempWeight = (pathDistanceBuffer[yValue * numNodes + k] + pathDistanceBuffer[k * numNodes + xValue]); \n"\
	"if (tempWeight < oldWeight){ \n"\
        "pathDistanceBuffer[yValue * numNodes + xValue] = tempWeight; \n"\
" } \n"\
"} \n"\
	"\n";

void printSolution(int dist[][DATA_SIZE])
{
            int i, j;
    for (i = 0; i < DATA_SIZE; i++)
    {
        for ( j = 0; j < DATA_SIZE; j++)
        {
            if (dist[i][j] == INF)
                printf("%7s", "INF");
            else
                seqfile << dist[i][j] <<" ";
        }
        seqfile << endl;
    }
}


void floydWarshall (int dist[][DATA_SIZE])
{
    int i, j, k;

    for (k = 0; k < DATA_SIZE; k++)
    {
        for (i = 0; i < DATA_SIZE; i++)
        {
            for (j = 0; j < DATA_SIZE; j++)
            {
                if (dist[i][k] + dist[k][j] < dist[i][j])
                    dist[i][j] = dist[i][k] + dist[k][j];
            }
        }
    }
	endy = clock();
    printSolution(dist);
}




int main(void)
{
cl_context context;
cl_context_properties properties[3];
cl_kernel kernel;
cl_command_queue command_queue;
cl_program program;
cl_int err;
cl_uint num_of_platforms=0;
cl_platform_id platform_id;
cl_device_id device_id;
cl_uint num_of_devices=0;
cl_mem path_dis_buffer, path_buffer;
outfile.open("parallel_result.txt");
seqfile.open("sequential_result.txt");
size_t global[2];
size_t local[2];
int block_size = 4;
int i, num_passes;


int path_dis_mat[DATA_SIZE*DATA_SIZE];
int path_mat[DATA_SIZE*DATA_SIZE];

int seq_dis_mat[DATA_SIZE][DATA_SIZE];

for(i=0; i<DATA_SIZE*DATA_SIZE; i++){
	path_dis_mat[i] = rand()%500 + 1; //it generates random number between 1-500
}

for(i=0; i<DATA_SIZE; i++)
{
	for(int j=0; j<DATA_SIZE; j++)
	{
		seq_dis_mat[i][j] = path_dis_mat[i*DATA_SIZE + j];
	}
}

for(cl_int i = 0; i < DATA_SIZE; ++i)
{
    for(cl_int j = 0; j < i; ++j)
        {
            path_mat[i * DATA_SIZE + j] = i;
            path_mat[j * DATA_SIZE + i] = j;
        }
    path_mat[i * DATA_SIZE + i] = i;
}

if(clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS)
{
	printf("Unable to get platform id\n");
	return 1;
}


if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
{
printf("Unable to get device_id\n");
return 1;
}

properties[0]= CL_CONTEXT_PLATFORM;
properties[1]= (cl_context_properties) platform_id;
properties[2]= 0;

context = clCreateContext(properties,1,&device_id,NULL,NULL,&err);

command_queue = clCreateCommandQueue(context, device_id, 0, &err);

program = clCreateProgramWithSource(context,1,(const char **) &ProgramSource, NULL, &err);

if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
{
printf("Error building program\n");
return 1;
}

kernel = clCreateKernel(program, "parallel_fw", &err);

path_dis_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * DATA_SIZE * DATA_SIZE, NULL, NULL);
path_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * DATA_SIZE * DATA_SIZE, NULL, NULL);

clEnqueueWriteBuffer(command_queue, path_dis_buffer, CL_TRUE, 0, sizeof(int) * DATA_SIZE * DATA_SIZE, path_dis_mat, 0, NULL, NULL);
clEnqueueWriteBuffer(command_queue, path_buffer, CL_TRUE, 0, sizeof(int) * DATA_SIZE * DATA_SIZE, path_mat, 0, NULL, NULL);


int temp = DATA_SIZE;
clSetKernelArg(kernel, 0, sizeof(cl_mem), &path_dis_buffer);
clSetKernelArg(kernel, 1, sizeof(cl_mem), &path_buffer);
clSetKernelArg(kernel, 2, sizeof(int), &temp);
clSetKernelArg(kernel, 3, sizeof(int), &temp);
global[0] = DATA_SIZE;
global[1] = DATA_SIZE;
local[0] = block_size;
local[1] = block_size;
num_passes = DATA_SIZE;
start = clock();
for(i=0; i<num_passes; i++)
{
    clSetKernelArg(kernel, 3, sizeof(int), &i);
    clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, NULL);
    clFlush(command_queue);
}

clFinish(command_queue);
endy = clock();
clEnqueueReadBuffer(command_queue, path_dis_buffer, CL_TRUE, 0, sizeof(int) *DATA_SIZE * DATA_SIZE, path_dis_mat, 0, NULL, NULL);
clEnqueueReadBuffer(command_queue, path_buffer, CL_TRUE, 0, sizeof(int) * DATA_SIZE * DATA_SIZE, path_mat, 0, NULL, NULL);


printf("output: ");

for(i=0; i<DATA_SIZE; i++)
{
//printf("%d ",path_dis_mat[i]);
	for(int j=0; j<DATA_SIZE; j++){
		outfile << path_dis_mat[i*DATA_SIZE + j] << " ";
	}
	outfile << endl;
}

double time_taken = ((double) (endy - start)) / CLK_TCK;
cout<<"Time taken is :"<<time_taken<<" Seconds"<<endl;
outfile<<"Time taken to run the code is: "<<(double)time_taken<<" Seconds"<<endl;


start = clock();
floydWarshall(seq_dis_mat);
//endy = clock();
time_taken = ((double) (endy - start)) / CLK_TCK;
seqfile<<"Time taken to run the code is: "<<(double)time_taken<<" Seconds"<<endl;

clReleaseMemObject(path_dis_buffer);
clReleaseMemObject(path_buffer);
clReleaseProgram(program);
clReleaseKernel(kernel);
clReleaseCommandQueue(command_queue);
clReleaseContext(context);
return 0;
}