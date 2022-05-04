// Copyright 2019 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SIMULATOR_BASIC_H_
#define SIMULATOR_BASIC_H_

#include <complex>
#include <cstdint>
#include <functional>
#include <vector>
#include <iostream>
#include <fstream>

#include "simulator.h"
#include "statespace_basic.h"
#include "zfp-develop/include/zfp.h"
#include "zfp-develop/cfp/include/cfparray.h"
#include "zfp-develop/cfp/include/cfparray1d.h"
#include "zfp-develop/cfp/include/cfparray1f.h"
#include "zfp-develop/cfp/include/cfparray2d.h"
#include "zfp-develop/cfp/include/cfparray2f.h"
#include "zfp-develop/cfp/include/cfparray3d.h"
#include "zfp-develop/cfp/include/cfparray3f.h"
#include "zfp-develop/cfp/include/cfparray4d.h"
#include "zfp-develop/cfp/include/cfparray4f.h"
#include "zfp-develop/cfp/include/cfpheader.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "zfp-develop/array/zfparray1.h"
#include "zfp-develop/array/zfparray2.h"
#include "zfp-develop/array/zfparray3.h"

static int test_counter = 0;

namespace qsim {

/**
 * Quantum circuit simulator without vectorization.
 */
template <typename For, typename FP = float>
class SimulatorBasic final : public SimulatorBase {
 public:
  using StateSpace = StateSpaceBasic<For, FP>;
  using State = typename StateSpace::State;
  using fp_type = typename StateSpace::fp_type;

  template <typename... ForArgs>
  explicit SimulatorBasic(ForArgs&&... args) : for_(args...) {}

  /**
   * Applies a gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyGate(const std::vector<unsigned>& qs,
                 const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      ApplyGateH<1>(qs, matrix, state);
      break;
    case 2:
      ApplyGateH<2>(qs, matrix, state);
      break;
    case 3:
      ApplyGateH<3>(qs, matrix, state);
      break;
    case 4:
      ApplyGateH<4>(qs, matrix, state);
      break;
    case 5:
      ApplyGateH<5>(qs, matrix, state);
      break;
    case 6:
      ApplyGateH<6>(qs, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Applies a controlled gate using non-vectorized instructions.
   * @param qs Indices of the qubits affected by this gate.
   * @param cqs Indices of control qubits.
   * @param cvals Bit mask of control qubit values.
   * @param matrix Matrix representation of the gate to be applied.
   * @param state The state of the system, to be updated by this method.
   */
  void ApplyControlledGate(const std::vector<unsigned>& qs,
                           const std::vector<unsigned>& cqs, uint64_t cvals,
                           const fp_type* matrix, State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    if (cqs.size() == 0) {
      ApplyGate(qs, matrix, state);
      return;
    }

    switch (qs.size()) {
    case 1:
      ApplyControlledGateH<1>(qs, cqs, cvals, matrix, state);
      break;
    case 2:
      ApplyControlledGateH<2>(qs, cqs, cvals, matrix, state);
      break;
    case 3:
      ApplyControlledGateH<3>(qs, cqs, cvals, matrix, state);
      break;
    case 4:
      ApplyControlledGateH<4>(qs, cqs, cvals, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }
  }

  /**
   * Computes the expectation value of an operator using non-vectorized
   * instructions.
   * @param qs Indices of the qubits the operator acts on.
   * @param matrix The operator matrix.
   * @param state The state of the system.
   * @return The computed expectation value.
   */
  std::complex<double> ExpectationValue(const std::vector<unsigned>& qs,
                                        const fp_type* matrix,
                                        const State& state) const {
    // Assume qs[0] < qs[1] < qs[2] < ... .

    switch (qs.size()) {
    case 1:
      return ExpectationValueH<1>(qs, matrix, state);
      break;
    case 2:
      return ExpectationValueH<2>(qs, matrix, state);
      break;
    case 3:
      return ExpectationValueH<3>(qs, matrix, state);
      break;
    case 4:
      return ExpectationValueH<4>(qs, matrix, state);
      break;
    case 5:
      return ExpectationValueH<5>(qs, matrix, state);
      break;
    case 6:
      return ExpectationValueH<6>(qs, matrix, state);
      break;
    default:
      // Not implemented.
      break;
    }

    return 0;
  }

  /**
   * @return The size of SIMD register if applicable.
   */
  static unsigned SIMDRegisterSize() {
    return 1;
  }

  static void encode(void* out, State& state, size_t numBlocks){
    
    zfp_stream* zfp;   // compressed stream
      bitstream* stream; // bit stream to write to or read from
      size_t* offset;    // per-block bit offset in compressed stream
      float* ptr;       // pointer to block being processed
      size_t bufsize;    // byte size of uncompressed storage
      size_t zfpsize;    // byte size of compressed stream
      uint minbits;      // min bits per block
      uint maxbits;      // max bits per block
      uint maxprec;      // max precision
      int minexp;        // min bit plane encoded
      uint bits;         // size of compressed block
      uint c;
      uint blocks = 1;
      float a[(int)(2*pow(2.0,state.num_qubits()))];
      double* buffer = (double*) malloc(sizeof(a));
      //double* buffer = (double*) malloc(2*sizeof(states));

      // maintain offset to beginning of each variable-length block
      //offset = (size_t*) malloc(blocks * sizeof(size_t));

      // associate bit stream with same storage as input
      bufsize = sizeof(*buffer);
      stream = stream_open(buffer, bufsize);

      // allocate meta data for a compressed stream
      zfp = zfp_stream_open(stream);

      // set tolerance for fixed-accuracy mode
      zfp_stream_set_accuracy(zfp, 1e3);

      // set maxbits to guard against prematurely overwriting the input
      zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
      maxbits = 4 * 4 * sizeof(buffer) * CHAR_BIT;
      zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp);

      // compress one block at a time in sequential order
      ptr = state.get();
      for (c = 0; c < (2*pow(2.0, state.num_qubits())); c++) {
        //printf("c: %d\n", c);
        //offset[c] = stream_wtell(stream);
        //printf("state vector value: %f\n", (ptr+c));
        bits = zfp_encode_block_float_1(zfp, ptr+c);
        //printf("Compression success!\n");
        if (!bits) {
          fprintf(stderr, "compression failed\n");
          return;
        }
        //printf("block #%u offset=%4u size=%4u\n", c, (uint)offset[c], bits);
      }
      // important: flush any buffered compressed bits
      stream_flush(stream);

      // print out size
      zfpsize = stream_size(stream);
      printf("compressed %u bytes to %u bytes\n", (uint)bufsize, (uint)zfpsize);


    /*unsigned minbits = 1024;
    unsigned maxbits = minbits;
    unsigned maxprec = ZFP_MAX_PREC;
    int minexp = ZFP_MIN_EXP;
    double states[(int)(2*pow(2.0, state.num_qubits()))];
    size_t bufsize = numBlocks*sizeof(states);
    void* buffer = malloc(bufsize);
    bitstream* stream = stream_open(buffer, bufsize);
    zfp_stream* zfp = zfp_stream_open(stream);
    zfp_stream_set_params(zfp,minbits,maxbits,maxprec,minexp);
    size_t bits = 0;
    for(size_t c=0; c<numBlocks; c++){
      bits = zfp_encode_block_float_1(zfp,state.get());
    }
    bits += stream_flush(stream);
    size_t bytes = (bits+7)/8;
    memcpy(out, buffer, bytes);
    stream_close(stream);
    zfp_stream_close(zfp);
    free(buffer);
    return bytes;*/
  }

  static void decode(void* in, State& state, size_t numBlocks){
      zfp_stream* zfp;   // compressed stream
      bitstream* stream; // bit stream to write to or read from
      size_t* offset;    // per-block bit offset in compressed stream
      float* ptr;       // pointer to block being processed
      size_t bufsize;    // byte size of uncompressed storage
      size_t zfpsize;    // byte size of compressed stream
      uint minbits;      // min bits per block
      uint maxbits;      // max bits per block
      uint maxprec;      // max precision
      int minexp;        // min bit plane encoded
      uint bits;         // size of compressed block
      uint c;


      zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp);
    /* decompress one block at a time in reverse order */
    ptr = state.get();
    for (c = 0; c < numBlocks; c++) {
      printf("c: %d\n", c);
      if (!zfp_decode_block_float_1(zfp, ptr)) {
        fprintf(stderr, "decompression failed\n");
        return;
      }
    }

    /* clean up */
    zfp_stream_close(zfp);
    stream_close(stream);
    /*unsigned minbits = 1024;
    unsigned maxbits = minbits;
    unsigned maxprec = ZFP_MAX_PREC;
    int minexp = ZFP_MIN_EXP;
    double states[(int)(2*pow(2.0, state.num_qubits()))];

    size_t bufsize = numBlocks*sizeof(states);
    bitstream *stream = stream_open((void*)in, bufsize);
    zfp_stream *zfp = zfp_stream_open(stream);
    zfp_stream_set_params(zfp,minbits,maxbits,maxprec,minexp);
    size_t bits = 0;
    for(size_t c = 0; c<numBlocks; c++){
      zfp_decode_block_float_1(zfp,state.get());
    }
    stream_flush(stream);
    stream_close(stream);
    zfp_stream_close(zfp);*/
  }

  static void compressArr(float* arr, size_t nx, zfp_bool decompress){
    
    int status = 0;    /* return value: 0 = success */
    zfp_type type;     /* array scalar type */
    zfp_field* field;  /* array meta data */
    zfp_stream* zfp;   /* compressed stream */
    void* buffer;      /* storage for compressed stream */
    size_t bufsize;    /* byte size of compressed buffer */
    bitstream* stream; /* bit stream to write to or read from */
    size_t zfpsize;    /* byte size of compressed stream */

    /* allocate meta data for the 1D array a[nx]*/
    type = zfp_type_float;
    field = zfp_field_1d(arr, type, nx);

    /* allocate meta data for a compressed stream */
    zfp = zfp_stream_open(NULL);

    /* set compression mode and parameters via one of four functions */
    /*  zfp_stream_set_reversible(zfp); */
    /*zfp_stream_set_rate(zfp, 12, type, zfp_field_dimensionality(field), zfp_false);*/
    /*  zfp_stream_set_precision(zfp, precision); */
    zfp_stream_set_accuracy(zfp, 1e-3);

    /* allocate buffer for compressed data */
    bufsize = zfp_stream_maximum_size(zfp, field);
    buffer = malloc(bufsize);

    /* associate bit stream with allocated buffer */
    stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    /* compress or decompress entire array */
    if (decompress) {
      size_t wHeaderSize = zfp_read_header(zfp, field, ZFP_HEADER_FULL);

      /* read compressed stream and decompress and output array */
      zfpsize = fread(buffer, 1, bufsize, fopen("compStream.bin", "r"));

      if (!zfp_decompress(zfp, field)) {
        fprintf(stderr, "decompression failed\n");
        status = EXIT_FAILURE;
      }
      else{
        fwrite(arr, sizeof(double), zfp_field_size(field, NULL), fopen("compStream.bin", "w"));
      }
    }
    else {
      size_t wHeaderSize = zfp_write_header(zfp, field, ZFP_HEADER_FULL);

      /* compress array and output compressed stream */
      zfpsize = zfp_compress(zfp, field);
      
      if (!zfpsize) {
        fprintf(stderr, "compression failed\n");
        status = EXIT_FAILURE;
      }
      else
        fwrite(buffer, 1, zfpsize, fopen("compStream.bin", "w"));
    }

    /* clean up */
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    free(buffer);
    //free(arr);
  }

  static float* populateVector(State& state){
    float* arr = new float[(int)(2*pow(2.0,state.num_qubits()))];
    memcpy(arr, state.get(), (2*pow(2.0,state.num_qubits())));
    /*for(int i = 0; i < 2*pow(2.0,state.num_qubits()); i++){
      //printf("Giving statevector value %f to vector\n", *(state.get()+i));
      //tmp.push_back(*(state.get()+i));
      arr[i] = *(state.get()+i);
    }*/
    return arr;
  }

    // qs = Indices of the qubits affected by this gate.
    // matrix = Matrix representation of the gate to be applied.
    // state = state vector
 private:
  template <unsigned H>
  void ApplyGateH(const std::vector<unsigned>& qs,
                  const fp_type* matrix, State& state) const {
    // f = function for seqfor to run
    // n = num_threads
    // m = current_thread
    // i = current_iteration
    // v = gate_matrix
    // ms = table of masks
    // xss = table of offset indices
    // rstate = state_vector
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;
      fp_type rn, in;
      //fp_type rs[hsize], is[hsize];

      //zfp::array1<fp_type> rn, in;
      zfp::array1<fp_type> rs(hsize, 0.0);
      zfp::array1<fp_type> is(hsize, 0.0);

      /*uint dims = 1;
      zfp_type type = zfp_type_float;
      zfp_field* field = zfp_field_1d(&rstate, type, hsize);

      zfp_stream* zfp = zfp_stream_open(NULL);

      zfp_stream_set_rate(zfp, 0, type, dims, 0);
      zfp_stream_set_precision(zfp, 0);
      zfp_stream_set_accuracy(zfp, 0);

      size_t bufsize = zfp_stream_maximum_size(zfp, field);
      uchar* buffer = new uchar[bufsize];

      bitstream* stream = stream_open(buffer, bufsize);
      zfp_stream_set_bit_stream(zfp, stream);

      size_t size = zfp_compress(zfp, field);*/

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      //zfp_stream_rewind(zfp);
      //size = zfp_decompress(zfp, field);

      // p0 = pointer to start index in state vector
      auto p0 = rstate + 2 * ii;


      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = *(p0 + xss[k]);
        is[k] = *(p0 + xss[k] + 1);
      }

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn += rs[l] * v[j] - is[l] * v[j + 1];
          in += rs[l] * v[j + 1] + is[l] * v[j];

          j += 2;
        }

        *(p0 + xss[k]) = rn;
        *(p0 + xss[k] + 1) = in;
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    /*if (test_counter < 6) {
        printf("test #%d\n", test_counter);
        printf("writing pre buffers to file...\n");
        char file_name[20];
        sprintf(file_name, "test%d_pre.bin", test_counter);
        FILE *test_file = fopen(file_name, "wb");
        unsigned local_h = H;
        uint64_t local_num_qubits = state.num_qubits();
        if (test_file) {
            //fwrite(&local_h, sizeof(unsigned), 1, test_file);
            //fwrite(&local_num_qubits, sizeof(uint64_t), 1, test_file);
            //fwrite(&size, sizeof(uint64_t), 1, test_file);
            //fwrite(ms, sizeof(uint64_t), H + 1, test_file);
            //fwrite(xss, sizeof(uint64_t), 1 << H, test_file);
            //fwrite(matrix, sizeof(fp_type), (1 << H) * (1 << H), test_file);
            fwrite(state.get(), sizeof(fp_type), 2 * (1 << state.num_qubits()), test_file);
            printf("finished writing buffers...\n");
        }
        fclose(test_file);
        //unsigned h_read;
        //uint64_t size_read;
        //uint64_t q_read;
        //uint64_t ms_read[H + 1];
        //uint64_t xss_read[1 << H];
        //fp_type matrix_read[(1 << H) * (1 << H)];
        fp_type state_read[2 * 1 << state.num_qubits()];
        FILE *read_test_file = fopen(file_name, "rb");
        if (read_test_file) {
            //fread(&h_read, sizeof(unsigned), 1, read_test_file);
            //fread(&q_read, sizeof(uint64_t), 1, read_test_file);
            //fread(&size_read, sizeof(uint64_t), 1, read_test_file);
            //fread(ms_read, sizeof(uint64_t), H + 1, read_test_file);
            //fread(xss_read, sizeof(uint64_t), 1 << H, read_test_file);
            //fread(matrix_read, sizeof(fp_type), (1 << H) * (1 << H), read_test_file);
            fread(state_read, sizeof(fp_type), 2 * (1 << state.num_qubits()), read_test_file);
            fclose(read_test_file);
        }
        //printf("wrote h: %u\n", local_h);
        //printf("read  h: %u\n", h_read);
        //printf("wrote size: %u\n", size);
        //printf("read  size: %u\n", size_read);
        //printf("wrote qubits: %u\n", state.num_qubits());
        //printf("read  qubits: %u\n", q_read);
        //printf("wrote ms[0]: %u\n", ms[0]);
        //printf("read  ms[0]: %u\n", ms_read[0]);
        printf("wrote state[0]: %f\n", state.get()[0]);
        printf("read  state[0]: %f\n", state_read[0]);
        //printf("wrote xss[0]: %u\n", xss[0]);
        //printf("read  xss[0]: %u\n", xss_read[0]);
    }*/
      
      /*void* buffer = malloc((4)*sizeof(float));
      size_t bufsize = (4)*sizeof(buffer);

      bitstream* stream = stream_open(buffer, bufsize);

      zfp_stream* zfp = zfp_stream_open(stream);

      zfp_stream_set_accuracy(zfp, 0.0);

      zfp_stream_params(zfp, 0, 0, 0, 0);
      uint maxbits = (4) * sizeof(&buffer) * CHAR_BIT;
      zfp_stream_set_params(zfp, 0, maxbits, 0, 0);

      uint bits;
      auto* ptr = state.get();
      for (int i = 0; i < (1<<H); i++) {
        bits = zfp_encode_block_float_2(zfp, ptr);
        if (!bits) {
          fprintf(stderr, "compression failed\n");
        }
        ptr += xss[i];
      }*/
      /*uint64_t i;
      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      float* p0 = state.get() + 2 * ii;*/

      //zfp::array2<double> states(2, 1<<H, 0.0);
      //double states[(int)(2*pow(2.0, state.num_qubits()))];

      
      
      /*zfp_stream* zfp;   // compressed stream
      bitstream* stream; // bit stream to write to or read from
      size_t* offset;    // per-block bit offset in compressed stream
      float* ptr;       // pointer to block being processed
      size_t bufsize;    // byte size of uncompressed storage
      size_t zfpsize;    // byte size of compressed stream
      uint minbits;      // min bits per block
      uint maxbits;      // max bits per block
      uint maxprec;      // max precision
      int minexp;        // min bit plane encoded
      uint bits;         // size of compressed block
      uint c;
      uint blocks = 1;
      //double* buffer = (double*) malloc(blocks*4*4*sizeof(double));
      double* buffer = (double*) malloc(2*sizeof(states));

      // maintain offset to beginning of each variable-length block
      //offset = (size_t*) malloc(blocks * sizeof(size_t));

      // associate bit stream with same storage as input
      bufsize = sizeof(*buffer);
      stream = stream_open(buffer, bufsize);

      // allocate meta data for a compressed stream
      zfp = zfp_stream_open(stream);

      // set tolerance for fixed-accuracy mode
      zfp_stream_set_accuracy(zfp, 1e3);

      // set maxbits to guard against prematurely overwriting the input
      zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
      maxbits = 4 * 4 * sizeof(buffer) * CHAR_BIT;
      zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp);

      // compress one block at a time in sequential order
      ptr = p0;
      for (c = 0; c < (2*pow(2.0, state.num_qubits())); c++) {
        //printf("c: %d\n", c);
        //offset[c] = stream_wtell(stream);
        //printf("state vector value: %f\n", (ptr+c));
        bits = zfp_encode_block_float_1(zfp, ptr+c);
        //printf("Compression success!\n");
        if (!bits) {
          fprintf(stderr, "compression failed\n");
          return;
        }
        //printf("block #%u offset=%4u size=%4u\n", c, (uint)offset[c], bits);
      }
      // important: flush any buffered compressed bits
      stream_flush(stream);

      // print out size
      zfpsize = stream_size(stream);
      printf("compressed %u bytes to %u bytes\n", (uint)bufsize, (uint)zfpsize);

      // decompress one block at a time in reverse order
      for (c = (2*pow(2.0, state.num_qubits())); c--;) {
        //stream_rseek(stream, offset[c]);
        if (!zfp_decode_block_float_1(zfp, ptr+c)) {
          fprintf(stderr, "decompression failed\n");
        }
      }

      // clean up
      zfp_stream_close(zfp);
      stream_close(stream);
      free(offset);
      */
      // print out size
      //zfpsize = stream_size(stream);
      //printf("compressed %u bytes to %u bytes\n", (uint)bufsize, (uint)zfpsize);


      
      /*uint dims = 1;
      zfp_type type = zfp_type_float;
      zfp_field* field = zfp_field_1d(&states, type, sizeof(2*pow(2.0,state.num_qubits())));

      float *buf = (float *)malloc(sizeof(zfp_type_size(field->type)) * field->nx);

      zfp_field_set_pointer(field, buf);

      zfp_stream* zfp = zfp_stream_open(NULL);

      zfp_stream_set_rate(zfp, 0, type, dims, 0);
      zfp_stream_set_precision(zfp, 0);
      zfp_stream_set_accuracy(zfp, 1e-3);

      size_t bufsize = zfp_stream_maximum_size(zfp, field);
      uchar* buffer = new uchar[bufsize];

      bitstream* stream = stream_open(buffer, bufsize);
      zfp_stream_set_bit_stream(zfp, stream);
      zfp_stream_rewind(zfp);

      if(zfp_write_header(zfp, field, ZFP_HEADER_FULL)==0){
        printf("Header write failed");
      }

      printf("State before compression: %d\n", state.get());

      size_t zfpSize = zfp_compress(zfp, field);

      printf("State after compression: %d\n", state.get());

      zfp_stream_rewind(zfp);

      if(zfp_read_header(zfp, field, ZFP_HEADER_FULL)==0){
        printf("Header read failed");
      }

      if(field->nx == 0) printf("Error");

      size_t zfpUncompressed = zfp_decompress(zfp, field);
      
      printf("State after decompression: %d\n", state.get());

      if(!zfpUncompressed){
        printf("Decompression failed");
      }

      printf("Compressed %d into %d and uncompressed %d into %d\n", (uint)bufsize, (uint)zfpSize, (uint)zfpSize, (uint)zfpUncompressed);
    */

    //printf("Printing statevector values: \n");
    //for (unsigned k = 0; k < (2*pow(2.0, state.num_qubits())); ++k) {
      //if((k%10)==0)
      //printf("%f, ", *(state.get()+k)); //Prints statevector value at index k
      //states[k] = *(p0+k);
      //printf("real: %f, ", *(p0+xss[k]));
      //states(0,k) = *(p0+xss[k]);
      //printf("imaginary: %f\n", *(p0+xss[k]+1));
      //states(1,k) = *(p0+xss[k]+1);
    //}
    //printf("\n");

    //printf("Populating vector...\n");
    /*if(test_counter==0){
      float* arr = populateVector(state);
      printf("\n");
      printf("Printing original arr values...\n");
      for(int c = 0; c < (2*pow(2.0,state.num_qubits())); c++){
        printf("%f", arr[c]);
      }
      compressArr(arr, (2*pow(2.0,state.num_qubits())), false);
      //compressArr(arr, (2*pow(2.0,state.num_qubits())), true);
      printf("\n");
      printf("Printing compressed arr values...\n");
      for(int c = 0; c < (2*pow(2.0,state.num_qubits())); c++){
        printf("%f", arr[c]);
      }
    }*/
    
    /*
    printf("\n");
    printf("Array initial size: %d", sizeof(arr));
    printf("\n");
    printf("Compressing array...\n");*/
    //printf("Arr size: %d\n", sizeof(arr));
    //compressArr(arr, (2*pow(2.0,state.num_qubits())), false);
    //printf("Arr size after compression: %d\n", sizeof(arr));
    /*printf("Printing compressed arr values...\n");
    for(int c = 0; c < (2*pow(2.0,state.num_qubits())); c++){
      printf("%f", arr[c]);
    }
    printf("\n");
    printf("Compressed array size: %d", sizeof(arr));
    printf("\n");*/
    //compressArr(arr, (2*pow(2.0,state.num_qubits())), true);
    //printf("Arr size after decompression: %d\n", sizeof(arr));
    /*printf("Printing decompressed arr values...\n");
    for(int c = 0; c < (2*pow(2.0,state.num_qubits())); c++){
      printf("%f", arr[c]);
    }
    printf("\n");*/
    

    //printf("Compressing state vector...\n");
    //void* buf = malloc(2*pow(2.0,state.num_qubits()));
    //encode(buf, state, 1);
    //printf("Compression success! Decompressing...\n");

    //printf("Printing statevector values: \n");
    //for (unsigned k = 0; k < (2*pow(2.0, state.num_qubits())); ++k) {
      //if((k%10)==0)
      //printf("%f, ", *(state.get()+k)); //Prints statevector value at index k
      //states[k] = *(p0+k);
      //printf("real: %f, ", *(p0+xss[k]));
      //states(0,k) = *(p0+xss[k]);
      //printf("imaginary: %f\n", *(p0+xss[k]+1));
      //states(1,k) = *(p0+xss[k]+1);
    //}
    //printf("\n");

    //decode(buf, state, 1);
    //printf("Decompression success! Running...\n");
    //printf("Running...\n");
    printf("Test counter: %d\n", test_counter);
    std::ofstream myfile;
    
    if(test_counter == 0){
      myfile.open("svDump.csv");
      myfile <<"test_counter == 0";
      for(int c = 0; c < (2*pow(2.0,state.num_qubits())); c++){
        //printf("%f\n", *(state.get()+c));
        if((c%10000)==0){
          myfile << std::endl;
        }
        myfile <<*(state.get()+c) << ";";
        
      }
      myfile << std::endl;
      myfile << std::endl;
    }
    else if(test_counter == 10){
      myfile.open("svDump.csv", std::ofstream::app);
      myfile <<"test_counter == 10";
      for(int c = 0; c < (2*pow(2.0,state.num_qubits())); c++){
        //printf("%f\n", *(state.get()+c));
        if((c%10000)==0){
          myfile << std::endl;
        }
        myfile <<*(state.get()+c) << ";";
        
      }
      myfile << std::endl;
      myfile << std::endl;
    }
    else if(test_counter == 100){
      myfile.open("svDump.csv", std::ofstream::app);
      myfile <<"test_counter == 100";
      for(int c = 0; c < (2*pow(2.0,state.num_qubits())); c++){
        //printf("%f\n", *(state.get()+c));
        if((c%10000)==0){
          myfile << std::endl;
        }
        myfile <<*(state.get()+c) << ";";
        
      }
      myfile << std::endl;
      myfile << std::endl;
    }

    if(test_counter>100){
      myfile.close();
    }
    

    for_.Run(size, f, matrix, ms, xss, state.get());

    //compressArr(arr, (2*pow(2.0,state.num_qubits())), false);
    
    
    

    //zfp_stream_rewind(zfp);
    //printf("Compressed %d into %d\n", (uint)zfpUncompressed, zfp_compress(zfp, field));

    /*if (test_counter < 6) {
        printf("writing post buffers to file...\n");
        char file_name[20];
        sprintf(file_name, "test%d_post.bin", test_counter);
        FILE *test_file = fopen(file_name, "wb");
        unsigned local_h = H;
        if (test_file) {
            fwrite(state.get(), sizeof(fp_type), 2 * (1 << state.num_qubits()), test_file);
            printf("finished writing buffers...\n");
        }

        fclose(test_file);
    }*/
    test_counter++;
  }

  template <unsigned H>
  void ApplyControlledGateH(const std::vector<unsigned>& qs,
                            const std::vector<unsigned>& cqs,
                            uint64_t cvals, const fp_type* matrix,
                            State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                uint64_t cvalsh, uint64_t cmaskh, fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      fp_type rn, in;
      zfp::array1<fp_type> rs(hsize, 0.0);
      zfp::array1<fp_type> is(hsize, 0.0);

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      if ((ii & cmaskh) == cvalsh) {
        auto p0 = rstate + 2 * ii;

        for (unsigned k = 0; k < hsize; ++k) {
          rs[k] = *(p0 + xss[k]);
          is[k] = *(p0 + xss[k] + 1);
        }

        uint64_t j = 0;

        for (unsigned k = 0; k < hsize; ++k) {
          rn = rs[0] * v[j] - is[0] * v[j + 1];
          in = rs[0] * v[j + 1] + is[0] * v[j];

          j += 2;

          for (unsigned l = 1; l < hsize; ++l) {
            rn += rs[l] * v[j] - is[l] * v[j + 1];
            in += rs[l] * v[j + 1] + is[l] * v[j];

            j += 2;
          }

          *(p0 + xss[k]) = rn;
          *(p0 + xss[k] + 1) = in;
        }
      }
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    auto m = GetMasks7(state.num_qubits(), qs, cqs, cvals);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    for_.Run(size, f, matrix, ms, xss, m.cvalsh, m.cmaskh, state.get());
  }

  template <unsigned H>
  std::complex<double> ExpectationValueH(const std::vector<unsigned>& qs,
                                         const fp_type* matrix,
                                         const State& state) const {
    auto f = [](unsigned n, unsigned m, uint64_t i, const fp_type* v,
                const uint64_t* ms, const uint64_t* xss,
                const fp_type* rstate) {
      constexpr unsigned hsize = 1 << H;

      fp_type rn, in;
      zfp::array1<fp_type> rs(hsize, 0.0);
      zfp::array1<fp_type> is(hsize, 0.0);

      uint64_t ii = i & ms[0];
      for (unsigned j = 1; j <= H; ++j) {
        i *= 2;
        ii |= i & ms[j];
      }

      auto p0 = rstate + 2 * ii;

      for (unsigned k = 0; k < hsize; ++k) {
        rs[k] = *(p0 + xss[k]);
        is[k] = *(p0 + xss[k] + 1);
      }

      double re = 0;
      double im = 0;

      uint64_t j = 0;

      for (unsigned k = 0; k < hsize; ++k) {
        rn = rs[0] * v[j] - is[0] * v[j + 1];
        in = rs[0] * v[j + 1] + is[0] * v[j];

        j += 2;

        for (unsigned l = 1; l < hsize; ++l) {
          rn += rs[l] * v[j] - is[l] * v[j + 1];
          in += rs[l] * v[j + 1] + is[l] * v[j];

          j += 2;
        }

        re += rs[k] * rn + is[k] * in;
        im += rs[k] * in - is[k] * rn;
      }

      return std::complex<double>{re, im};
    };

    uint64_t ms[H + 1];
    uint64_t xss[1 << H];

    FillIndices<H>(state.num_qubits(), qs, ms, xss);

    unsigned n = state.num_qubits() > H ? state.num_qubits() - H : 0;
    uint64_t size = uint64_t{1} << n;

    using Op = std::plus<std::complex<double>>;
    return for_.RunReduce(size, f, Op(), matrix, ms, xss, state.get());
  }

  For for_;
};

}  // namespace qsim

#endif  // SIMULATOR_BASIC_H_
