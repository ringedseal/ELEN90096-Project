#include "srcnn.h"

void conv2_tile(
    ftmap_t input_ftmap[N1][H][W],
    param_t conv2_weights[N2][N1][F2][F2],
    param_t conv2_biases[N2],
    int      k_channel,
    int      i_height,
    int      j_width,
    int      tn_eff,
    int      th_eff,
    int      tw_eff,
	ftmap_t  tile_out[Tn2][Th2][Tw2]
){
	// on chip tile buffer
	ftmap_t out_tile[Tn2][Th2][Tw2];
#pragma HLS BIND_STORAGE variable=out_tile type=ram_t2p impl=bram
#pragma HLS DEPENDENCE  variable=out_tile intra false inter false

	// output tile bias
	for (int tn = 0; tn < tn_eff; ++tn) {
		for (int th = 0; th < th_eff; ++th){
			for (int tw = 0; tw < tw_eff; ++tw)
				out_tile[tn][th][tw] = conv2_biases[k_channel + tn];
		}
	}

	// build input block and accumulate (no need halo)
	for (int c0 = 0; c0 < N1; c0 += Tc2){
		const int tc_eff = (c0 + Tc2 <= N1) ? Tc2 : (N1 - c0);
		ftmap_t in_tile[Tc2][Th2][Tw2];
#pragma HLS BIND_STORAGE variable=in_tile  type=ram_t2p impl=bram
#pragma HLS DEPENDENCE  variable=in_tile  intra false inter false
#pragma HLS ARRAY_PARTITION variable=in_tile dim=1 factor=2

#pragma HLS UNROLL factor=2
		for (int tc = 0; tc < tc_eff; ++tc){
			for (int th = 0; th < th_eff; ++th) {
				for (int tw = 0; tw < tw_eff; ++tw) {
					in_tile[tc][th][tw] = input_ftmap[c0 + tc][i_height + th][j_width + tw];
				}
			}
		}
		//MAC
		for (int tn = 0; tn < tn_eff; ++tn) {
		  for (int th = 0; th < th_eff; ++th) {

#pragma HLS PIPELINE II=1
		    for (int tw = 0; tw < tw_eff; ++tw) {
		      ftmap_t acc = out_tile[tn][th][tw];
#pragma HLS UNROLL factor=2
		      for (int tc = 0; tc < Tc2; ++tc) {
		          if (tc < tc_eff) {
		              const param_t w = conv2_weights[k_channel + tn][c0 + tc][0][0];
		              acc += w * in_tile[tc][th][tw];
		          }
		      }
		      out_tile[tn][th][tw] = acc;
		    }
		  }
		}
	}

	// write back and add ReLU
	for (int tn = 0; tn < tn_eff; ++tn) {
		for (int th = 0; th < th_eff; ++th) {
			for (int tw = 0; tw < tw_eff; ++tw) {
				ftmap_t v = out_tile[tn][th][tw];
				tile_out[tn][th][tw] = (v > 0.f ? v : 0.f);
			}
		}
	}
}

void conv3_tile(
    ftmap_t  input_tile[N2][Th3+F3-1][Tw3+F3-1],
    param_t  conv3_weights[N3][N2][F3][F3],
    param_t  conv3_biases[N3],
    int      k_channel,
    int      i_height,
    int      j_width,
    int      tn_eff,
    int      th_eff,
    int      tw_eff,
    ftmap_t  output_ftmap[N3][H][W]
){
#pragma HLS INLINE off
	int padding_size = (F3-1)/2;
	ftmap_t out_tile[Tn3][Th3][Tw3];

//#pragma HLS ARRAY_PARTITION variable=out_tile dim=1 cyclic factor=4
#pragma HLS BIND_STORAGE variable=out_tile type=ram_t2p impl=bram
#pragma HLS DEPENDENCE variable=out_tile intra false inter false

	// output tile bias
	for (int tn = 0; tn < tn_eff; ++tn) {

		for (int th = 0; th < th_eff; ++th){

			for (int tw = 0; tw < tw_eff; ++tw){
#pragma HLS PIPELINE II=1
			    out_tile[tn][th][tw] = conv3_biases[k_channel + tn];
			}
		}
	}

	// Traverse the input channel block (different with conv1 since N2 is not 0)
	// Perform a "row buffer convolution" on each tc sub-block, and then add the accumulated result back to out_tile
	for (int c0 = 0; c0 < N2; c0 += Tc3){
		const int tc_eff = (c0 + Tc3 <= N2) ? Tc3 : (N2 - c0);

		// row buffer: F3 rows, each row width = (tw_eff + F3 - 1)
		ftmap_t lb [Tc3][F3][Tw3 + F3 - 1];
//#pragma HLS ARRAY_PARTITION variable=lb dim=1 factor=2 cyclic
//#pragma HLS ARRAY_PARTITION variable=lb dim=2 complete
//#pragma HLS ARRAY_PARTITION variable=lb dim=3 complete
//#pragma HLS DEPENDENCE variable=lb intra false inter false
#pragma HLS BIND_STORAGE variable=lb type=ram_t2p impl=bram
#pragma HLS DEPENDENCE variable=lb intra false inter false

		// Initialize the first F3 lines of the row buffer
        for (int tc = 0; tc < tc_eff; ++tc) {
            for (int kh = 0; kh < F3; ++kh) {

                for (int tw = 0; tw < tw_eff + F3 - 1; ++tw) {
#pragma HLS PIPELINE II=1

                    lb[tc][kh][tw] = input_tile[c0 + tc][kh][tw];
                }
            }
        }

        //!!!Main scan: for each row of tile th!!!
        for (int th = 0; th < th_eff; ++th) {

            ftmap_t win[Tc3][F3][F3]; // window Tc3xF3xF3 , first expand spatial part (kh, kw),then accumulate on tc

//#pragma HLS ARRAY_PARTITION variable=win complete
//#pragma HLS DEPENDENCE variable=win intra false inter false
#pragma HLS BIND_STORAGE variable=win type=ram_s2p impl=bram
#pragma HLS DEPENDENCE variable=win intra false inter false
            // get first window from lb
            for (int tc = 0; tc < tc_eff; ++tc) {

                for (int kh = 0; kh < F3; ++kh) {

                    for (int kw = 0; kw < F3; ++kw) {
#pragma HLS PIPELINE II=1
                        win[tc][kh][kw] = lb[tc][kh][kw];
                    }
                }
            }

            // Convolution on each pixel column tw in this row
            for (int tw = 0; tw < tw_eff; ++tw) {
#pragma HLS PIPELINE II=1
            	for (int tn = 0; tn < tn_eff; ++tn) {
//#pragma HLS UNROLL factor=2
            		ftmap_t acc = out_tile[tn][th][tw];
                    for (int tc = 0; tc < tc_eff; ++tc) {
//#pragma HLS UNROLL factor=2
                        for (int kh = 0; kh < F3; ++kh) {
//#pragma HLS UNROLL
                            for (int kw = 0; kw < F3; ++kw) {
//#pragma HLS UNROLL
                                const param_t w = conv3_weights[k_channel + tn][c0 + tc][kh][kw];
                                acc += w * win[tc][kh][kw];
                            }
                        }
                    }
                    out_tile[tn][th][tw] = acc;
            	}

            	// slip the window, move right one column; fill new column from row buffer
                if (tw + 1 < tw_eff) {
                    for (int tc = 0; tc < tc_eff; ++tc) {

                        for (int kh = 0; kh < F3; ++kh) {

                            // shift left
                            for (int kw = 0; kw < F3 - 1; ++kw) {

                                win[tc][kh][kw] = win[tc][kh][kw + 1];
                            }

                            // new column from lb
                            win[tc][kh][F3 - 1] = lb[tc][kh][tw + F3];
                        }
                    }
                }
            }

            //after this line end, scroll lb down one line
            if (th + F3 < th_eff + F3 - 1) {
                for (int tc = 0; tc < tc_eff; ++tc) {
                    for (int tw = 0; tw < tw_eff + F3 - 1; ++tw) {
#pragma HLS PIPELINE II=1

                        // shift row buffer up
                        for (int kh = 0; kh < F3 - 1; ++kh) {

                            lb[tc][kh][tw] = lb[tc][kh + 1][tw];
                        }

                        lb[tc][F3 - 1][tw] = input_tile[c0 + tc][th + F3][tw];
                    }
                }
            }

        } // th end
	} // c0 end

	// write back
    for (int tn = 0; tn < tn_eff; ++tn) {
        for (int th = 0; th < th_eff; ++th) {

            for (int tw = 0; tw < tw_eff; ++tw) {
#pragma HLS PIPELINE II=1
                output_ftmap[k_channel + tn][i_height + th][j_width + tw] = out_tile[tn][th][tw];
            }
        }
    }
}

void srcnn(ftmap_t input_ftmap[N0][H][W],
                 param_t conv1_weights[N1][N0][F1][F1],
                 param_t conv1_biases[N1],
                 param_t conv2_weights[N2][N1][F2][F2],
                 param_t conv2_biases[N2],
                 param_t conv3_weights[N3][N2][F3][F3],
                 param_t conv3_biases[N3],
                 ftmap_t output_ftmap[N3][H][W])
{
	 static ftmap_t conv1_out[N1][H][W];
	 conv1(input_ftmap, conv1_weights, conv1_biases, conv1_out);

	 const int pad = (F3 - 1) / 2;

	 for (int i_height = 0; i_height < H; i_height += Th3) {
		 const int th_eff = (i_height + Th3 <= H) ? Th3 : (H - i_height);

		 for (int j_width = 0; j_width < W; j_width += Tw3) {
			 const int tw_eff = (j_width + Tw3 <= W) ? Tw3 : (W - j_width);

	         // build in_tile_for_conv3 with halo
	         ftmap_t in_tile_for_conv3[N2][Th3 + F3 - 1][Tw3 + F3 - 1];

	         for (int oc = 0; oc < N2; ++oc){ //conv2 output channel (also conv3 input channel)
	        	 for (int th = 0; th < th_eff + F3 - 1; ++th){
	        		 int global_y = i_height + th - pad;
	                 if (global_y < 0)       global_y = 0;
	                 if (global_y >= H)      global_y = H - 1;


	                 for (int tw = 0; tw < tw_eff + F3 - 1; ++tw){

	                	 int global_x = j_width + tw - pad;
	                     if (global_x < 0)    global_x = 0;
	                     if (global_x >= W)   global_x = W - 1;

	                     ftmap_t acc = conv2_biases[oc];
	                     for (int ic = 0; ic < N1; ++ic) {
	                    	 const param_t w = conv2_weights[oc][ic][0][0];
	                         acc += w * conv1_out[ic][global_y][global_x];
	                     }

	                     //ReLU
	                     if (acc < 0) acc = 0;

	                     in_tile_for_conv3[oc][th][tw] = acc;
	                 }
	             }
	         }

	         conv3_tile(
	             in_tile_for_conv3,
	             conv3_weights,
	             conv3_biases,
	             0,
	             i_height,
	             j_width,
	             Tn3,
	             th_eff,
	             tw_eff,
	             output_ftmap
	         );
		 }
	 }
}
