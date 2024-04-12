#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <sys/stat.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
//#define uint64_t unsigned long long

long long negabinary2binary(unsigned long long x){
    return (x ^0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
}

void decode_block(unsigned long long * streams, int n, int num_bitplanes, unsigned long long * data, int bitplane_size, int blockid) {
    for(int k=num_bitplanes - 1; k>=0; k--){
        unsigned long long bitplane_index = num_bitplanes - 1 - k;
        unsigned long long bitplane_value = streams[bitplane_index * bitplane_size + blockid];
        //if (blockid == 0)   printf("bitplane_value = %llu\n", bitplane_value);
        for (int i=0; i<n; i++){
            data[i] = data[i] | (((bitplane_value >> i) & 1u) << k);
        }
    }
}
double * NegaDecode(unsigned long long * streams, int n, int exp, int num_bitplanes) {
            
    int block_size = 64;
    int bitplane_size = n / block_size + 1;
    //printf("bitplane_size = %d\n", bitplane_size);
    //printf("level_exp = %d\n", exp);
    //printf("Decode number of bitplanes = %d\n", num_bitplanes);
    int starting_bitplane = 0;
    double * data = (double *) malloc(n * sizeof(double));
    int offset = 0;
    if(num_bitplanes == 0){
        memset(data, 0, n * sizeof(double));
        return data;
    }
    // leave room for negabinary format
    exp += 2;
    unsigned long long * int_data_buffer = (unsigned long long *)calloc(block_size, sizeof(unsigned long long));
    // decode
    int ending_bitplane = starting_bitplane + num_bitplanes;
    if(ending_bitplane % 2 == 0){
        for(int i=0; i<n - block_size; i+=block_size){
            memset(int_data_buffer, 0, block_size * sizeof(unsigned long long));
            decode_block(streams, block_size, num_bitplanes, int_data_buffer, bitplane_size, i/block_size);
            for(int j=0; j<block_size; j++){
                *(data + offset) = ldexp((double) negabinary2binary(int_data_buffer[j]), - ending_bitplane + exp);
                offset += 1;
            }
        }
        {
            int rest_size = n % block_size;
            if(rest_size == 0) rest_size = block_size;
            memset(int_data_buffer, 0, rest_size * sizeof(unsigned long long));
            decode_block(streams, rest_size, num_bitplanes, int_data_buffer, bitplane_size, bitplane_size - 1);
            for(int j=0; j<rest_size; j++){
                *(data + offset) = ldexp((double) negabinary2binary(int_data_buffer[j]), - ending_bitplane + exp);
                offset += 1;
            }
        }
    }
    else{
        for(int i=0; i<n - block_size; i+=block_size){
            memset(int_data_buffer, 0, block_size * sizeof(unsigned long long));
            decode_block(streams, block_size, num_bitplanes, int_data_buffer, bitplane_size, i/block_size);
            for(int j=0; j<block_size; j++){
                *(data + offset) = - ldexp((double) negabinary2binary(int_data_buffer[j]), - ending_bitplane + exp);
                offset += 1;
            }
        }
        {
            int rest_size = n % block_size;
            if(rest_size == 0) rest_size = block_size;
            memset(int_data_buffer, 0, rest_size * sizeof(unsigned long long));
            decode_block(streams, rest_size, num_bitplanes, int_data_buffer, bitplane_size, bitplane_size - 1);
            for(int j=0; j<rest_size; j++){
                *(data + offset) = - ldexp((double) negabinary2binary(int_data_buffer[j]), - ending_bitplane + exp);
                offset += 1;
            }
        }
    }
    return data;
}


