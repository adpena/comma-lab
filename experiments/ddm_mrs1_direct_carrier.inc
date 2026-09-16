# Direct carrier interpretation. The entropy decoders are unchanged; intermediate
# Huffman/Rice serialization is omitted because its decoded arrays are consumed.
def direct_carrier_layout(body):
    if len(body) < 142:
        raise ValueError('truncated packed carrier metadata')
    basis_bits = int.from_bytes(body[:3], 'little')
    coefficient_bits = int.from_bytes(body[3:6], 'little')
    basis_end = 142 + (basis_bits + 7) // 8
    coefficient_end = basis_end + (coefficient_bits + 7) // 8
    if coefficient_end > len(body):
        raise ValueError('truncated packed carrier streams')
    return basis_bits, basis_end, coefficient_end


def decode_carrier(body):
    """Decode counted spatial bases and their signed 12-bit AR(1) trajectory."""
    basis_bits, basis_end, coefficient_end = direct_carrier_layout(body)
    symbols = basis_decode_basis_arith(body[142:basis_end], basis_bits)
    basis_codes = (symbols >> 1) ^ -(symbols & 1)
    factors = body[102] + archive_unpack_unsigned(body[103:114], 12, 7)
    biases = archive_unpack_unsigned(body[114:123], 12, 6)
    biases = np.where(biases >= 32, biases - 64, biases).astype(np.int16)
    parameters = body[139] + archive_unpack_unsigned(body[140:142], 12, 1)
    symbols = coefficients_cabac_decode(body[basis_end:coefficient_end], parameters)
    residuals = (symbols >> 1) ^ -(symbols & 1)
    codes = np.empty((600, 12), dtype=np.int32)
    codes[0] = residuals[0]
    for frame in range(1, 600):
        products = codes[frame - 1].astype(np.int64) * factors.astype(np.int64)
        rounded = np.where(products >= 0, (products + 128) // 256,
                           -((-products + 128) // 256))
        prediction = ((rounded + biases + 2048) & 4095) - 2048
        codes[frame] = ((prediction + residuals[frame] + 2048) & 4095) - 2048
    scales = np.frombuffer(body[6:54], dtype='<f4').copy()
    coefficient_scales = np.frombuffer(body[54:102], dtype='<f4').copy()
    basis = torch.from_numpy(basis_codes.astype(np.int8)).reshape(12, 3, 24, 32).float()
    basis = basis * torch.from_numpy(scales)[:, None, None, None]
    coefficients = torch.from_numpy(codes).float() * torch.from_numpy(coefficient_scales)[None]
    selector, _ = overlay_split_selector_compensation(
        archive_SPARSE_SELECTOR_PREFIX + body[coefficient_end:])
    selector_decode_selector(selector)
    return basis, coefficients, selector


def archive_decode_rx1_models(outer):
    """Keep the packed carrier bytes; decode them directly when loading models."""
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = archive_RX1_MODEL_HEADER.unpack_from(outer)
    model_end = archive_RX1_MODEL_HEADER.size + hpac_bytes + semantic_bytes + carrier_bytes
    offset = archive_RX1_MODEL_HEADER.size
    hpac_stream = outer[offset:offset + hpac_bytes]
    offset += hpac_bytes
    semantic_stream = outer[offset:offset + semantic_bytes]
    offset += semantic_bytes
    carrier_stream = outer[offset:offset + carrier_bytes]
    hpac = archive_lzma.decompress(hpac_stream, format=archive_lzma.FORMAT_XZ) if codec == archive_RX1_CODEC_XZ else archive_decompress_brotli(hpac_stream)
    if reserved & archive_RC1_RESERVED_HPAC_ADAPTIVE:
        hpac = archive_ck2_uninterleave_planes(hpac)
    semantic = archive_decompress_brotli(semantic_stream)
    if reserved & archive_CK2_RESERVED_SEMANTIC_PLANE2:
        semantic = archive_ck2_uninterleave_planes(semantic)
    required = archive_RR5_RESERVED_ARITH_BASIS | archive_DX2_RESERVED_CABAC_COEFFICIENTS
    if reserved & required != required:
        raise ValueError('expected arithmetic basis and binary-coded coefficients')
    carrier = archive_decompress_brotli(carrier_stream)
    _, _, coefficient_end = direct_carrier_layout(carrier)
    _, compensation = overlay_split_selector_compensation(
        archive_SPARSE_SELECTOR_PREFIX + carrier[coefficient_end:])
    return semantic, carrier, hpac, compensation, outer[model_end:], outer[:model_end]


def read_models(archive_path):
    parts = archive_read_residual_archive(archive_path)
    model = render_SemanticTokenRenderer(96)
    semantic_blob = weight_mixer_restore_semantic(parts.semantic_blob, model.state_dict())
    model.load_state_dict(weights_unpack_variant_semantic_or_none(semantic_blob, model.state_dict()), strict=True)
    basis, coefficients, selector_blob = decode_carrier(parts.carrier_blob)
    return parts, model, basis, coefficients, selector_blob
