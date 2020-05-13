def get_in_out_channels(net):
    in_channels = out_channels = 0
    for i in range(len(net)):
        try:
            in_channels = net[i].in_channels
            break
        except AttributeError:
            pass
    else:
        raise AttributeError("network has no conv2d layers so in_channels and out_channels can not be found")

    # find out_channels
    for i in range(len(net)-1,-1,-1):
        try:
            out_channels = net[i].out_channels
            break
        except AttributeError:
            pass

    return in_channels, out_channels

def num_pixels(x):
    num_elements = x.nelement()
    B, C, D = x.size(0), x.size(1), x.size(2)
    return num_elements / B / C / D

def unsqueeze3D(input, upscale_factor=(2,2,1)):
    '''
    [:, C*r1*r2*r3, H, W, L] -> [:, C, H*r1, W*r2, L*r3]
    '''
    batch_size, in_channels, in_height, in_width, in_lambda = input.size()
    out_channels = in_channels // (upscale_factor[0] *
                                   upscale_factor[1] *
                                   upscale_factor[2])

    out_height = in_height * upscale_factor[0]
    out_width = in_width * upscale_factor[1]
    out_lambda = in_lambda * upscale_factor[2]

    input_view = input.contiguous().view(batch_size, out_channels,
        upscale_factor[0], upscale_factor[1], upscale_factor[2],
        in_height, in_width, in_lambda
    )

    output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width, out_lambda)


def squeeze3D(input, downscale_factor=(2,2,1)):
    '''
    [:, C, H*r1, W*r2, L*r3] -> [:, C*r1*r2*r3, H, W, L]
    '''
    batch_size, in_channels, in_height, in_width, in_lambda = input.size()
    out_channels = in_channels * (downscale_factor[0] *
                                  downscale_factor[1] *
                                  downscale_factor[2])

    out_height, residue_height  = divmod(in_height, downscale_factor[0])
    out_width, residue_width = divmod(in_width, downscale_factor[1])
    out_lambda, residue_lambda  = divmod(in_lambda, downscale_factor[2])

    assert residue_height == 0, f"squeezed to much; height is not divideble by {downscale_factor[0]}"
    assert residue_width == 0, f"squeezed to much; width is not divideble by {downscale_factor[1]}"
    assert residue_lambda == 0, f"squeezed to much; number of wavelengths is not divideble by {downscale_factor[2]}"

    input_view = input.contiguous().view(batch_size, in_channels,
                                         out_height, downscale_factor[0],
                                         out_width, downscale_factor[1],
                                         out_lambda, downscale_factor[2]
    )

    output = input_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return output.view(batch_size, out_channels, out_height, out_width, out_lambda)
