
//from: http://stackoverflow.com/questions/24267280/efficiently-find-mimimum-of-large-array-using-opencl

__kernel void findMinValue(__global float * myArray, __global double * mins, __global int * arraysize,__global int * elToWorkOn,__global int * dummy){
int gid = get_global_id(0);
int lid = get_local_id(0);
int groupID = get_group_id(0);
int lsize = get_local_size(0);
int gsize = get_global_id(0);
int minloc = 0;
int arrSize = *arraysize;
int elPerGroup = *elToWorkOn;
float mymin = INFINITY;


__local float lmins[128];
//initialize local memory
*(lmins + lid) = INFINITY;
__local int lminlocs[128];

//this private value will reduce global memory access in the for loop (temp = *(myArray + i);)
float temp;

//ofset and target of the for loop
int offset = elPerGroup*groupID + lid;
int target = elPerGroup*(groupID + 1);

//prevent that target<arrsize (may happen due to rounding errors or arrSize not a multiple of elPerGroup
target = min(arrSize, target);

//find minimum for the kernel
//offset is different for each lid, leading to sequential memory access
if (offset < arrSize){
    for (int i = offset; i < target; i += lsize){
        temp = *(myArray + i);
        if (temp < mymin){
            mymin = temp;
            minloc = i;
        }
    }

    //store kernel minimum in local memory
    *(lminlocs + lid) = minloc;
    *(lmins + lid) = mymin;

    //find work group minimum (reduce global memory accesses)
    lsize = lsize >> 1;
    while (lsize > 0){
        if (lid < lsize){
            if (*(lmins + lid)> *(lmins + lid + lsize)){
                *(lmins + lid) = *(lmins + lid + lsize);
                *(lminlocs + lid) = *(lminlocs + lid + lsize);
            }
        }
        lsize = lsize >> 1;
    }
}
//write group minimum to global buffer
if (lid == 0){
    *(mins + groupID * 2 + 0) = *(lminlocs + 0);
    *(mins + groupID * 2 + 1) = *(lmins + 0);
}
}
