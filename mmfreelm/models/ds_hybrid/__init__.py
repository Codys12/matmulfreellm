# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from mmfreelm.models.ds_hybrid.configuration_ds_hybrid import DSHybridConfig
from mmfreelm.models.ds_hybrid.modeling_ds_hybrid import DSHybridForCausalLM, DSHybridModel

AutoConfig.register(DSHybridConfig.model_type, DSHybridConfig)
AutoModel.register(DSHybridConfig, DSHybridModel)
AutoModelForCausalLM.register(DSHybridConfig, DSHybridForCausalLM)


__all__ = ['DSHybridConfig', 'DSHybridForCausalLM', 'DSHybridModel']
