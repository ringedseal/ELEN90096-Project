#include "srcnn.h"

// implements conv1 layer of SRCNN
void conv1(ftmap_t input_ftmap[N0][H][W],
           param_t conv1_weights[N1][N0][F1][F1],
           param_t conv1_biases[N1],
           ftmap_t output_ftmap[N1][H][W])
{
#pragma HLS ARRAY_PARTITION variable=conv1_weights complete dim=3
#pragma HLS ARRAY_PARTITION variable=conv1_weights complete dim=4
#pragma HLS ARRAY_PARTITION variable=conv1_weights cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=conv1_biases  cyclic factor=4 dim=1
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
#pragma HLS ARRAY_PARTITION variable=out_tile cyclic factor=4 dim=1
#pragma HLS BIND_STORAGE variable=in_tile  type=ram_t2p impl=bram
#pragma HLS BIND_STORAGE variable=out_tile type=ram_t2p impl=bram
#pragma HLS DEPENDENCE  variable=in_tile  intra false inter false
#pragma HLS DEPENDENCE  variable=out_tile intra false inter false

            	// output tile bias
            	for (int tn = 0; tn < tn_eff; ++tn) {
#pragma HLS UNROLL factor=4
            		for (int th = 0; th < th_eff; ++th){
#pragma HLS PIPELINE II=1
            			for (int tw = 0; tw < tw_eff; ++tw)
            			    out_tile[tn][th][tw] = conv1_biases[k_channel + tn];
            		}
            	}

            	// build input halo (N0=1)
            	for (int th = 0; th < th_eff + F1 - 1; ++th) {
            		int ih = clamp(i_height + th - padding_size, 0, H - 1);
#pragma HLS PIPELINE II=1
            		for (int tw = 0; tw < tw_eff + F1 - 1; ++tw) {
            	        int jw = clamp(j_width + tw - padding_size, 0, W - 1);
            	        in_tile[0][th][tw] = input_ftmap[0][ih][jw];
            	    }
            	}

            	// Do convolution, no need clamp any more
            	for (int tn = 0; tn < tn_eff; ++tn) {
#pragma HLS UNROLL factor=4
            		// row buffering: store line F1, length of each line = tw_eff + F1 - 1
            		ftmap_t lb[F1][Tw1 + F1 - 1];
#pragma HLS ARRAY_PARTITION variable=lb complete dim=1
#pragma HLS BIND_STORAGE variable=lb type=ram_s2p impl=bram

            		//store the F1 lines of tile to the row buffer (include right halo)
				    for (int kh = 0; kh < F1; ++kh){
					    for (int tw = 0; tw < tw_eff + F1 - 1; ++tw){
						    lb[kh][tw] = in_tile[0][kh][tw];
					    }
				    }

				    for (int th = 0; th < th_eff; ++th) {
				    	ftmap_t win[F1][F1];	// window F1xF1
#pragma HLS ARRAY_PARTITION variable=win complete dim=0

				    	//initialize the previous F1 column of window
				    	for (int kh = 0; kh < F1; ++kh) {
#pragma HLS UNROLL
				    		for (int kw = 0; kw < F1; ++kw) {
#pragma HLS UNROLL
				    			win[kh][kw] = lb[kh][kw];
				    		}
				    	}

				    	for (int tw = 0; tw < tw_eff; ++tw) {
#pragma HLS PIPELINE II=1
				    		//calculate F1xF1 parallel MAC
				    		ftmap_t acc = out_tile[tn][th][tw];
//#pragma HLS UNROLL
				    		for (int kh = 0; kh < F1; ++kh) {
#pragma HLS UNROLL
				    			for (int kw = 0; kw < F1; ++kw) {
#pragma HLS UNROLL
				    				const param_t w = conv1_weights[k_channel + tn][0][kh][kw];
				    				acc += w * win[kh][kw];
				    			}
				    		}
				    		out_tile[tn][th][tw] = acc;

				    		//slip the window, move left one column; fill new column from row buffer
				    		if (tw + 1 < tw_eff) {
#pragma HLS UNROLL
				    			for (int kh = 0; kh < F1; ++kh) {
				    				// shift left
#pragma HLS UNROLL
				    				for (int kw = 0; kw < F1 - 1; ++kw) {
				    					win[kh][kw] = win[kh][kw + 1];
				    				}
				    			// new column: from the (tw + F1) column of row buffer
				    			win[kh][F1 - 1] = lb[kh][tw + F1];
				    			}
				    		}

				    	}
				    	// row buffer load the next line
				    	if (th + F1 < th_eff + F1 - 1){
#pragma HLS PIPELINE II=1
				    		for (int tw = 0; tw < tw_eff + F1 - 1; ++tw){
				    			for (int kh = 0; kh < F1 - 1; ++kh) {
				    				lb[kh][tw] = lb[kh + 1][tw];
				    			}
				    			lb[F1 - 1][tw] = in_tile[0][th + F1][tw];
				    		}
				    	}
				   }
            	}

                // write back and add ReLU
                for (int tn = 0; tn < tn_eff; ++tn) {
                    for (int th = 0; th < th_eff; ++th) {
#pragma HLS PIPELINE II=1
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
