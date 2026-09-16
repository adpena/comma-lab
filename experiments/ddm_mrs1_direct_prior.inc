def prior_weight_counts(model):
    """Count the coded weights in the architecture's existing masked rows."""
    rows = []
    for module in model.modules():
        if isinstance(module, prior_IntegerConv2d):
            mask = module.mask.to(bool).expand_as(module.weight)
            rows.extend(int(mask[index].sum().item()) for index in range(module.weight.shape[0]))
        elif isinstance(module, prior_IntegerLinear):
            rows.extend(int(row.numel()) for row in module.weight)
    return rows


def render_load_hpac(raw, device):
    model = prior_IntegerHPAC(num_pairs=render_N, num_classes=render_NUM_CLASSES, patch=render_HPAC_PATCH, delta=render_HPAC_DELTA, channels=render_HPAC_CHANNELS, frame_dim=render_HPAC_FILM_DIM, norm_mode='none', activation='relu', use_frame_scale=True, weight_bound=127, activation_bound=127, use_weight_scales=True, weight_exponent_min=-6, use_spm=True, use_norm_gates=False).eval()
    prior_io_deserialize_self_compressed(model, prior_mixer_restore_hpac(raw, prior_weight_counts(model)))
    return model.to(device)
