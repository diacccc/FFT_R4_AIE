#include <vector>
#include <complex>
#include <cmath>

template <typename INPUT_DATATYPE, typename OUTPUT_DATATYPE>
void fft_r4_stage1_ref(const std::vector<INPUT_DATATYPE>& input, 
                                std::vector<OUTPUT_DATATYPE>& output,
                                int fft_size) {
    // Radix-4 stage 1 reference implementation
    output.assign(fft_size * 2, 0); // Initialize output with zeros (complex numbers)

    const int RADIX = 4; 
    const int NUM_GROUPS = fft_size / RADIX;

    std::vector<std::complex<OUTPUT_DATATYPE>> stage1_raw(fft_size);

    for (int g = 0; g < NUM_GROUPS; ++g) {
        int idx0 = g * RADIX;
        int idx1 = idx0 + 1;
        int idx2 = idx0 + 2;
        int idx3 = idx0 + 3;

        // Load input values (complex numbers)
        std::complex<OUTPUT_DATATYPE> a(input[2*idx0], input[2*idx0 + 1]);
        std::complex<OUTPUT_DATATYPE> b(input[2*idx1], input[2*idx1 + 1]);
        std::complex<OUTPUT_DATATYPE> c(input[2*idx2], input[2*idx2 + 1]);
        std::complex<OUTPUT_DATATYPE> d(input[2*idx3], input[2*idx3 + 1]);

        // Compute radix-4 butterfly
        std::complex<OUTPUT_DATATYPE> y0 = a + b + c + d;
        std::complex<OUTPUT_DATATYPE> y1 = a - std::complex<OUTPUT_DATATYPE>(0, 1)*b - c + std::complex<OUTPUT_DATATYPE>(0, 1)*d;
        std::complex<OUTPUT_DATATYPE> y2 = a - b + c - d;
        std::complex<OUTPUT_DATATYPE> y3 = a + std::complex<OUTPUT_DATATYPE>(0, 1)*b - c - std::complex<OUTPUT_DATATYPE>(0, 1)*d;

        // Store output values
        stage1_raw[idx0] = y0;
        stage1_raw[idx1] = y1;
        stage1_raw[idx2] = y2;
        stage1_raw[idx3] = y3;
    }

    for (int g = 0; g < NUM_GROUPS; ++g) {
        for (int r = 0; r < RADIX; ++r) {
            int old_idx = g * RADIX + r;
            int new_idx = r * NUM_GROUPS + g;

            output[2*new_idx] = stage1_raw[old_idx].real();
            output[2*new_idx + 1] = stage1_raw[old_idx].imag();
        }
    }
    
    
}