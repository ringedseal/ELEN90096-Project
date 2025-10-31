#include "srcnn.h"

// implements conv1 layer of SRCNN
void conv1(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           ftmap_t output_ftmap[N1][H][W])
{
    // implement conv1 layer of SRCNN here
    int padding_size = (F1-1)/2;
    //Outer 3 loop for tile
    for (int k_channel = 0; k_channel<N1; k_channel += Tn1) {
    	const int tn_eff = (k_channel + Tn1 <= N1) ? Tn1 : (N1 - k_channel);

        for (int i_height = 0; i_height<H; i_height += Th1) {
        	const int th_eff = (i_height + Th1 <= H) ? Th1 : (H - i_height);

            for (int j_width = 0; j_width<W; j_width += Tw1) {
            	const int tw_eff = (j_width + Tw1 <= W) ? Tw1 : (W - j_width);

            	// on chip tile buffer
            	ftmap_t in_tile[1][Th1 + F1 - 1][Tw1 + F1 - 1];
            	ftmap_t out_tile[Tn1][Th1][Tw1];

            	// output tile bias
            	for (int tn = 0; tn < tn_eff; ++tn) {
            		for (int th = 0; th < th_eff; ++th){
            			for (int tw = 0; tw < tw_eff; ++tw)
            			    out_tile[tn][th][tw] = conv1_biases[k_channel + tn];
            		}
            	}

            	// build input halo (N0=1)
            	for (int th = 0; th < th_eff + F1 - 1; ++th) {
            		int ih = clamp(i_height + th - padding_size, 0, H - 1);

            		for (int tw = 0; tw < tw_eff + F1 - 1; ++tw) {
            	        int jw = clamp(j_width + tw - padding_size, 0, W - 1);
            	        in_tile[0][th][tw] = input_ftmap[0][ih][jw];
            	    }
            	}

            	// Do convolution, no need clamp any more
            	for (int tn = 0; tn < tn_eff; ++tn) {
            	  for (int th = 0; th < th_eff; ++th) {
#pragma HLS PIPELINE II=1
            	    for (int tw = 0; tw < tw_eff; ++tw) {
            	      ftmap_t acc = out_tile[tn][th][tw];
            	      for (int kh = 0; kh < F1; ++kh) {
            	        for (int kw = 0; kw < F1; ++kw) {
            	          const param_t w = conv1_weights[k_channel + tn][0][kh][kw];
            	          acc += w * in_tile[0][th + kh][tw + kw];
            	        }
            	      }
            	      out_tile[tn][th][tw] = acc;
            	    }
            	  }
            	}

                // write back and add ReLU
                for (int tn = 0; tn < tn_eff; ++tn) {
                    for (int th = 0; th < th_eff; ++th) {
                        for (int tw = 0; tw < tw_eff; ++tw) {
                            ftmap_t v = out_tile[tn][th][tw];
                            output_ftmap[k_channel + tn][i_height + th][j_width + tw] = (v > 0.f ? v : 0.f);
                        }
                    }
                }

            }
        }
    }
}
