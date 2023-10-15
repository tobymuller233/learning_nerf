import os
import imp
import ipdb


def make_network(cfg):
    module = cfg.network_module
    path = cfg.network_path
    # ipdb.set_trace()
    network = imp.load_source(module, path).Network()
    # ipdb.set_trace()
    return network
