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

template <typename INPUT_DATATYPE, typename OUTPUT_DATATYPE>
void fft_r4_stage2_ref(const std::vector<INPUT_DATATYPE>& input,
                       std::vector<OUTPUT_DATATYPE>& output,
                       int fft_size) {
    output.assign(fft_size * 2, OUTPUT_DATATYPE(0));

    const int RADIX = 4;
    const int NUM_GROUPS_STAGE1 = fft_size / RADIX;      // N/4
    const int NUM_GROUPS_STAGE2 = NUM_GROUPS_STAGE1 / 4; // N/16

    const double pi = std::acos(-1.0);

    // Raw stage-2 output before final permutation
    std::vector<std::complex<OUTPUT_DATATYPE>> stage2_raw(fft_size);

    // Input layout is assumed to be stage-1 output:
    // flat index = r0 * (N/4) + p1, where p1 = 4*g + r1
    for (int r0 = 0; r0 < RADIX; ++r0) {
        for (int g = 0; g < NUM_GROUPS_STAGE2; ++g) {
            int p0 = 4 * g + 0;
            int p1 = 4 * g + 1;
            int p2 = 4 * g + 2;
            int p3 = 4 * g + 3;

            int idx0 = r0 * NUM_GROUPS_STAGE1 + p0;
            int idx1 = r0 * NUM_GROUPS_STAGE1 + p1;
            int idx2 = r0 * NUM_GROUPS_STAGE1 + p2;
            int idx3 = r0 * NUM_GROUPS_STAGE1 + p3;

            std::complex<OUTPUT_DATATYPE> a(input[2 * idx0], input[2 * idx0 + 1]);
            std::complex<OUTPUT_DATATYPE> b(input[2 * idx1], input[2 * idx1 + 1]);
            std::complex<OUTPUT_DATATYPE> c(input[2 * idx2], input[2 * idx2 + 1]);
            std::complex<OUTPUT_DATATYPE> d(input[2 * idx3], input[2 * idx3 + 1]);

            auto twiddle = [&](int exp) -> std::complex<OUTPUT_DATATYPE> {
                double angle = -2.0 * pi * exp / static_cast<double>(fft_size);
                return {
                    static_cast<OUTPUT_DATATYPE>(std::cos(angle)),
                    static_cast<OUTPUT_DATATYPE>(std::sin(angle))
                };
            };

            // Stage-2 twiddles: W_N^(r0 * r1), r1 = 0,1,2,3
            a *= twiddle(4 * r0 * 0);
            b *= twiddle(4 * r0 * 1);
            c *= twiddle(4 * r0 * 2);
            d *= twiddle(4 * r0 * 3);

            // Radix-4 butterfly
            std::complex<OUTPUT_DATATYPE> y0 = a + b + c + d;
            std::complex<OUTPUT_DATATYPE> y1 = a - std::complex<OUTPUT_DATATYPE>(0, 1) * b
                                                 - c + std::complex<OUTPUT_DATATYPE>(0, 1) * d;
            std::complex<OUTPUT_DATATYPE> y2 = a - b + c - d;
            std::complex<OUTPUT_DATATYPE> y3 = a + std::complex<OUTPUT_DATATYPE>(0, 1) * b
                                                 - c - std::complex<OUTPUT_DATATYPE>(0, 1) * d;

            // Keep raw outputs contiguous within each r0 block
            stage2_raw[idx0] = y0;
            stage2_raw[idx1] = y1;
            stage2_raw[idx2] = y2;
            stage2_raw[idx3] = y3;
        }
    }

    // Final permutation to match your desired stage-2 layout:
    // old_idx = r0*(N/4) + (4*g + r1)
    // new_idx = g*(N/4) + r1*(N/16) + r0
    for (int r0 = 0; r0 < RADIX; ++r0) {
        for (int g = 0; g < NUM_GROUPS_STAGE2; ++g) {
            for (int r1 = 0; r1 < RADIX; ++r1) {
                int old_idx = r0 * NUM_GROUPS_STAGE1 + (4 * g + r1);
                int new_idx = g * NUM_GROUPS_STAGE1 + r1 * NUM_GROUPS_STAGE2 + r0;

                output[2 * new_idx]     = stage2_raw[old_idx].real();
                output[2 * new_idx + 1] = stage2_raw[old_idx].imag();
            }
        }
    }
}

template <typename INPUT_DATATYPE, typename OUTPUT_DATATYPE>
void fft_r4_stage3_ref(const std::vector<INPUT_DATATYPE>& input,
                                 std::vector<OUTPUT_DATATYPE>& output,
                                 int fft_size) {
    output.assign(fft_size * 2, OUTPUT_DATATYPE(0));

    const int RADIX = 4;
    const double pi = std::acos(-1.0);

    auto twiddle = [&](int exp) -> std::complex<OUTPUT_DATATYPE> {
        double angle = -2.0 * pi * exp / static_cast<double>(fft_size);
        return {
            static_cast<OUTPUT_DATATYPE>(std::cos(angle)),
            static_cast<OUTPUT_DATATYPE>(std::sin(angle))
        };
    };

    // After stage2:
    // input index = g * 16 + r1 * 4 + r0
    // g  = 0..3
    // r1 = 0..3
    // r0 = 0..3
    for (int r1 = 0; r1 < RADIX; ++r1) {
        for (int r0 = 0; r0 < RADIX; ++r0) {
            int idx0 = 0 * 16 + r1 * 4 + r0;
            int idx1 = 1 * 16 + r1 * 4 + r0;
            int idx2 = 2 * 16 + r1 * 4 + r0;
            int idx3 = 3 * 16 + r1 * 4 + r0;

            std::complex<OUTPUT_DATATYPE> a(input[2 * idx0], input[2 * idx0 + 1]);
            std::complex<OUTPUT_DATATYPE> b(input[2 * idx1], input[2 * idx1 + 1]);
            std::complex<OUTPUT_DATATYPE> c(input[2 * idx2], input[2 * idx2 + 1]);
            std::complex<OUTPUT_DATATYPE> d(input[2 * idx3], input[2 * idx3 + 1]);

            // The remaining twiddle depends on the combined previous frequency digit.
            // Depending on your chosen digit order, this is usually something like:
            int alpha = r1 * 4 + r0;

            a *= twiddle(alpha * 0);
            b *= twiddle(alpha * 1);
            c *= twiddle(alpha * 2);
            d *= twiddle(alpha * 3);

            std::complex<OUTPUT_DATATYPE> y0 = a + b + c + d;
            std::complex<OUTPUT_DATATYPE> y1 = a - std::complex<OUTPUT_DATATYPE>(0, 1) * b
                                                 - c + std::complex<OUTPUT_DATATYPE>(0, 1) * d;
            std::complex<OUTPUT_DATATYPE> y2 = a - b + c - d;
            std::complex<OUTPUT_DATATYPE> y3 = a + std::complex<OUTPUT_DATATYPE>(0, 1) * b
                                                 - c - std::complex<OUTPUT_DATATYPE>(0, 1) * d;

            // Natural-ish output order: k2 * 16 + r1 * 4 + r0
            output[2 * (0 * 16 + r1 * 4 + r0)]     = y0.real();
            output[2 * (0 * 16 + r1 * 4 + r0) + 1] = y0.imag();

            output[2 * (1 * 16 + r1 * 4 + r0)]     = y1.real();
            output[2 * (1 * 16 + r1 * 4 + r0) + 1] = y1.imag();

            output[2 * (2 * 16 + r1 * 4 + r0)]     = y2.real();
            output[2 * (2 * 16 + r1 * 4 + r0) + 1] = y2.imag();

            output[2 * (3 * 16 + r1 * 4 + r0)]     = y3.real();
            output[2 * (3 * 16 + r1 * 4 + r0) + 1] = y3.imag();
        }
    }
}