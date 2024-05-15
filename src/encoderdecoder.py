import numpy as np
import os
from scipy.stats import chisquare
import torch as th
import torch.nn.functional as F
import math
from src.imec import apply_random_mask, remove_random_mask, IMECEncoder, IMECDecoder
from src.meteor import METEOREncoder, METEORDecoder
from src.arithmetic import ArithmeticEncoder, ArithmeticDecoder
#from src.adg import ADGEncoder, ADGDecoder



def EncoderGen(cfg, private_message_bit, mask_cfg, medium, proc_seed):
    if cfg.method == "imec":
        encoder_input = apply_random_mask(private_message_bit, **mask_cfg)
        block_size = cfg.block_size  # in bits
        encoder = IMECEncoder(block_size=block_size, medium=medium,
                                clean_up_output=False,
                                seed=proc_seed,
                                mec_mode=cfg.mec_mode,
                                belief_entropy_threshold=cfg.imec_belief_entropy_threshold)
                                # medium needs to support the logit() function

        decoder = IMECDecoder(block_size=block_size,
                                n_chunks=int(math.ceil(len(private_message_bit) / block_size)),
                                medium=medium,
                                clean_up_output=False,
                                belief_entropy_threshold=cfg.imec_belief_entropy_threshold,
                                mec_mode=cfg.mec_mode)
    if cfg.method == "adg":
        encoder_input = apply_random_mask(private_message_bit, **mask_cfg)
        block_size = cfg.block_size  # in bits
        encoder = ADGEncoder(block_size=block_size, medium=medium,
                                clean_up_output=False,
                                seed=proc_seed,
                                mec_mode=cfg.mec_mode,
                                belief_entropy_threshold=cfg.imec_belief_entropy_threshold)
                                # medium needs to support the logit() function

        decoder = ADGDecoder(block_size=block_size,
                                n_chunks=int(math.ceil(len(private_message_bit) / block_size)),
                                medium=medium,
                                clean_up_output=False,
                                belief_entropy_threshold=cfg.imec_belief_entropy_threshold,
                                mec_mode=cfg.mec_mode)

    elif cfg.method == "meteor":
        print('using meteor')
        encoder_input = private_message_bit
        encoder = METEOREncoder(medium=medium,
                                seed=cfg.seed,
                                cleanup_output=False,
                                finish_sent=cfg.meteor_finish_sent,
                                precision=cfg.meteor_precision,
                                is_sort=cfg.meteor_is_sort,
                                **mask_cfg
                                )  # medium needs to support the logit() function

        decoder = METEORDecoder(medium=medium,
                                seed=proc_seed,
                                cleanup_output=False,
                                finish_sent=cfg.meteor_finish_sent,
                                precision=cfg.meteor_precision,
                                is_sort=cfg.meteor_is_sort,
                                **mask_cfg
                                )  # medium needs to support the logit() function

    elif cfg.method == "arithmetic":
        print('using arithmetic')
        encoder_input = private_message_bit
        encoder = ArithmeticEncoder(medium=medium,
                                seed=cfg.seed,
                                cleanup_output=False,
                                finish_sent=cfg.meteor_finish_sent,
                                precision=cfg.meteor_precision,
                                is_sort=cfg.meteor_is_sort,
                                **mask_cfg
                                )  # medium needs to support the logit() function

        decoder = ArithmeticDecoder(medium=medium,
                                seed=proc_seed,
                                cleanup_output=False,
                                finish_sent=cfg.meteor_finish_sent,
                                precision=cfg.meteor_precision,
                                is_sort=cfg.meteor_is_sort,
                                **mask_cfg
                                )  # medium needs to support the logit() function
    else:
        raise Exception("UNKNOWN method: {}".format(cfg.method))
    return(encoder, decoder, encoder_input)
