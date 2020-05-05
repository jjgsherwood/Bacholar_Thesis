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
