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

#pragma HLS BIND_STORAGE variable=out_tile type=ram_t2p impl=bram
#pragma HLS DEPENDENCE  variable=out_tile intra false inter false
            	// output tile bias
            	for (int tn = 0; tn < tn_eff; ++tn) {
            		for (int th = 0; th < th_eff; ++th){
            			for (int tw = 0; tw < tw_eff; ++tw)
            			    out_tile[tn][th][tw] = conv3_biases[k_channel + tn];
            		}
            	}

            	// build input halo and accumulate
				for (int c0 = 0; c0 < N2; c0 += Tc3){
					const int tc_eff = (c0 + Tc3 <= N2) ? Tc3 : (N2 - c0);
					ftmap_t in_tile[Tc3][Th3+F3-1][Tw3+F3-1];

#pragma HLS BIND_STORAGE variable=in_tile  type=ram_t2p impl=bram
#pragma HLS DEPENDENCE  variable=in_tile  intra false inter false

					for (int tc = 0; tc < tc_eff; ++tc){
						for (int th = 0; th < th_eff + F3 - 1; ++th) {
							int ih = clamp(i_height + th - padding_size, 0, H - 1);
							for (int tw = 0; tw < tw_eff + F3 - 1; ++tw) {
								int jw = clamp(j_width + tw - padding_size, 0, W - 1);
								in_tile[tc][th][tw] = input_ftmap[c0 + tc][ih][jw];
							}
						}
					}
					// MAC
					for (int tn = 0; tn < tn_eff; ++tn) {
					  for (int th = 0; th < th_eff; ++th) {
#pragma HLS PIPELINE II=1
					    for (int tw = 0; tw < tw_eff; ++tw) {

					      ftmap_t acc = out_tile[tn][th][tw];
//#pragma HLS UNROLL factor=TC_UNR
					      for (int tc = 0; tc < tc_eff; ++tc) {

//#pragma HLS UNROLL
					        for (int kh = 0; kh < F3; ++kh) {
//#pragma HLS UNROLL
					          for (int kw = 0; kw < F3; ++kw) {
					            const param_t w = conv3_weights[k_channel + tn][c0 + tc][kh][kw];
					            acc += w * in_tile[tc][th + kh][tw + kw];
					          }
					        }
					      }
					      out_tile[tn][th][tw] = acc;
					    }
					  }
					}
				}

                // write back (No ReLU)
                for (int tn = 0; tn < tn_eff; ++tn) {
                    for (int th = 0; th < th_eff; ++th) {
                        for (int tw = 0; tw < tw_eff; ++tw) {
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
