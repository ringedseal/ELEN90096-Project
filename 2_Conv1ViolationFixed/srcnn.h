#ifndef _SRCNN_H_
#define _SRCNN_H_

// image dimensions
#define W  255          // image width
#define H  255          // image height
#define UP 3            // upscaling factor

// CNN dimensions
#define N0 1            // input features (image channels)
#define N1 64           // conv1 output features
#define F1 9            // conv1 kernel size
#define N2 32           // conv2 output features
#define F2 1            // conv2 kernel size
#define N3 1            // conv3 output features
#define F3 5            // conv3 kernel size

// ==== Tiling parameters ====
#ifndef Th1
#define Th1  32    // tile output height
#endif
#ifndef Tw1
#define Tw1  32    // tile output width
#endif
#ifndef Tn1
#define Tn1 8     // conv1: output channel block (N1=64 so 8 channels for one block)
#endif

#ifndef Th2
#define Th2 64   // conv2: No halo, can be larger
#endif
#ifndef Tw2
#define Tw2 64
#endif
#ifndef Tc2
#define Tc2 16   // conv2: input channel block (N1=64 so 4 channels for one block)
#endif
#ifndef Tn2
#define Tn2 16   // conv2: output channel block (N2=32 so 2 channels for one block)
#endif

#ifndef Th3
#define Th3 32   // conv3
#endif
#ifndef Tw3
#define Tw3 32
#endif
#ifndef Tc3
#define Tc3 8    // conv3: input channel block (N2=32 so 4 channels for one block)
#endif
#ifndef Tn3
#define Tn3 1    // conv3: output channel block (N3=1 so 1 channel for one block)
#endif

// data types
typedef float ftmap_t;  // feature map
typedef float param_t;  // parameters

// implements end-to-end SRCNN
void srcnn(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           param_t conv2_weights[N2][N1][F2][F2],
           param_t conv2_biases[N2],
           param_t conv3_weights[N3][N2][F3][F3],
           param_t conv3_biases[N3],
           ftmap_t output_ftmap[N3][H][W]);

// implements first convolutional layer of SRCNN
void conv1(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           ftmap_t output_ftmap[N1][H][W]);

// Use Clamp to do the edge replicate
static inline int clamp(int x, int low_bound, int high_bound){
	return x<low_bound ? low_bound : (x>high_bound ? high_bound : x);
}

#endif /* _SRCNN_H_ */
