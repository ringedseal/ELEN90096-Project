#include "srcnn.h"

void conv2(ftmap_t input_ftmap[N1][H][W],
           param_t conv2_weights[N2][N1][F2][F2],
           param_t conv2_biases[N2],
           ftmap_t output_ftmap[N2][H][W])
{
#pragma HLS ARRAY_PARTITION variable=conv2_weights dim=2 factor=2
	//Kernel size is 1, so no need halo
	//Outer 3 loop for tile
	for (int k_channel = 0; k_channel<N2; k_channel += Tn2) {
		const int tn_eff = (k_channel + Tn2 <= N2) ? Tn2 : (N2 - k_channel);

		for (int i_height = 0; i_height<H; i_height += Th2) {
			const int th_eff = (i_height + Th2 <= H) ? Th2 : (H - i_height);

			for (int j_width = 0; j_width<W; j_width += Tw2) {
				const int tw_eff = (j_width + Tw2 <= W) ? Tw2 : (W - j_width);

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
							output_ftmap[k_channel + tn][i_height + th][j_width + tw] = (v > 0.f ? v : 0.f);
						}
					}
				}

			}
		}
	}
}

void conv3(ftmap_t input_ftmap[N2][H][W],
           param_t conv3_weights[N3][N2][F3][F3],
           param_t conv3_biases[N3],
           ftmap_t output_ftmap[N3][H][W])
{
	const int TC_UNR = 2;
    int padding_size = (F3-1)/2;

    //Outer 3 loop for tile
    for (int k_channel = 0; k_channel<N3; k_channel += Tn3) {
    	const int tn_eff = (k_channel + Tn3 <= N3) ? Tn3 : (N3 - k_channel);

        for (int i_height = 0; i_height<H; i_height += Th3) {
        	const int th_eff = (i_height + Th3 <= H) ? Th3 : (H - i_height);

            for (int j_width = 0; j_width<W; j_width += Tw3) {
            	const int tw_eff = (j_width + Tw3 <= W) ? Tw3 : (W - j_width);

            	// on chip tile buffer
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
                            int src_h = i_height + kh - padding_size;
                            src_h = clamp(src_h, 0, H - 1);

                            for (int tw = 0; tw < tw_eff + F3 - 1; ++tw) {
#pragma HLS PIPELINE II=1
                                int src_w = j_width + tw - padding_size;
                                src_w = clamp(src_w, 0, W - 1);

                                lb[tc][kh][tw] = input_ftmap[c0 + tc][src_h][src_w];
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

                                    // fill last row from input_ftmap
                                    int src_h_next = i_height + th + F3 - padding_size;
                                    src_h_next = clamp(src_h_next, 0, H - 1);

                                    int src_w = j_width + tw - padding_size;
                                    src_w = clamp(src_w, 0, W - 1);

                                    lb[tc][F3 - 1][tw] = input_ftmap[c0 + tc][src_h_next][src_w];
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
    // Implement end-to-end SRCNN here
	 static ftmap_t conv1_out[N1][H][W];
	 static ftmap_t conv2_out[N2][H][W];

	conv1(input_ftmap, conv1_weights, conv1_biases, conv1_out);

	conv2(conv1_out, conv2_weights, conv2_biases, conv2_out);

	conv3(conv2_out, conv3_weights, conv3_biases, output_ftmap);
}
