#version 450 core

// Workgroup size for matrix processing blocks
layout(local_size_x = 16, local_size_y = 16) in;

// Buffers for input matrices and output matrix
layout(std430, binding = 0) buffer MatrixABuffer {
    float matrix1[]; // Matrix A
};

layout(std430, binding = 1) buffer MatrixBBuffer {
    float matrix2[]; // Matrix B
};

layout(std430, binding = 2) buffer MatrixCBuffer {
    float result[]; // Resulting matrix C
};

// Uniforms to store matrix dimensions
layout(std140, binding = 3) uniform MatrixSizeInfo {
    ivec4 out_sizes; // Output matrix dimensions (rows, columns, unused, unused)
    ivec4 matrix1_sizes; // Matrix A dimensions (rows, columns, unused, unused)
    ivec4 matrix2_sizes; // Matrix B dimensions (rows, columns, unused, unused)
};

void main() {
    // Obtain global indices from invocation ID
    uint globalRow = gl_GlobalInvocationID.y;
    uint globalCol = gl_GlobalInvocationID.x;

    // Ensure indices are within the output matrix bounds
    if (globalRow >= uint(out_sizes.y) || globalCol >= uint(out_sizes.x)) {
        return;
    }

    // Calculate the value for this cell of the output matrix
    float sum = 0.0;
    for (int k = 0; k < matrix1_sizes.x; ++k) {
        // Compute linear indices for accessing matrix elements
        int matrix1Index = int(globalRow) * matrix1_sizes.x + k;
        int matrix2Index = k * matrix2_sizes.x + int(globalCol);

        // Perform element-wise multiplication and accumulate the sum
        sum += matrix1[matrix1Index] * matrix2[matrix2Index];
    }

    // Write the computed sum to the output matrix result
    int resultIndex = int(globalRow) * out_sizes.x + int(globalCol);
    result[resultIndex] = sum;
}
