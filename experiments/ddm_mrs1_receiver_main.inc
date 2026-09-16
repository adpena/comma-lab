# The archive contains the models; these names describe their fixed architecture.
from types import SimpleNamespace
from pathlib import Path
import hashlib
import sys
import torch

prior_namespace = SimpleNamespace(IntegerConv2d=prior_IntegerConv2d, IntegerLinear=prior_IntegerLinear)
render_namespace = SimpleNamespace(**{
    name.removeprefix('render_'): value for name, value in list(globals().items())
    if name.startswith('render_')
}, IntegerHPAC=prior_IntegerHPAC)


class TokenDecoder:
    """Decode the next class plane using only already decoded symbols."""

    def __init__(self, parts, device):
        self.device = device
        self.parts = parts
        prior_blob = prior_format_materialize_ihs1(parts.hpac_blob, render_namespace)
        self.model = render_load_hpac(prior_blob, device)
        self.sparse = sparse_SparseIntegerHPAC(self.model, render_EVAL_H, render_EVAL_W)
        inference_optimize_sparse_evaluator(self.sparse)
        self.corrector = miss_FreeCorrector(render_EVAL_H * render_EVAL_W)
        self.range_decoder = ArithmeticDecoder(parts.token_stream)
        self.mixer = geometry_mixer_LaneMixer(parts.tc1_weights)
        self.plans = []
        for mask in render_group_masks(device):
            positions = np.flatnonzero(mask.cpu().numpy().reshape(-1))
            self.plans.append((torch.from_numpy(positions).to(device), positions))
        self.previous = torch.zeros((1, render_EVAL_H, render_EVAL_W), dtype=torch.long, device=device)
        self.frame = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame == render_N:
            raise StopIteration
        index = torch.tensor([self.frame], dtype=torch.long, device=self.device)
        current = torch.zeros_like(self.previous)
        context = self.model.prepare_frame_context(index, self.previous)
        previous_field = None if self.frame == 0 else self.previous[0].cpu().to(torch.uint8).numpy()
        boundary = (np.full(render_EVAL_H * render_EVAL_W, 4, dtype=np.uint8)
                    if previous_field is None else archive_boundary_buckets(previous_field).reshape(-1))
        self.corrector.begin_frame(boundary)
        self.mixer.begin_frame()
        for group, (device_positions, positions) in enumerate(self.plans):
            logits = self.sparse.selected_logits(current, context, group).cpu().numpy()
            predicted = logits.argmax(axis=1).astype(np.int64)
            feature = boundary[positions].astype(np.int64) * 5 + predicted
            probability = archive_probability_table(logits + self.parts.table.values[feature], render_HPAC_LOGIT_PRECISION)
            state = self.corrector.group_state(probability, predicted, positions)
            coding = self.corrector.coding_row(state)
            partial = current[0].cpu().to(torch.uint8).numpy()
            coding = self.mixer.coding(coding, positions, partial, previous_field)
            symbols = self.range_decoder.decode(coding).astype(np.int64)
            self.mixer.observe(positions, symbols)
            self.corrector.observe(state, symbols)
            current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(self.device)
        tokens = current[0].cpu().to(torch.uint8)
        self.corrector.end_frame(tokens.numpy().reshape(-1))
        self.mixer.end_frame(tokens.numpy(), previous_field)
        self.previous = current
        self.frame += 1
        return tokens


def read_models(archive_path):
    parts = archive_read_residual_archive(archive_path)
    model = render_SemanticTokenRenderer(96)
    semantic_blob = weight_mixer_restore_semantic(parts.semantic_blob, model.state_dict())
    model.load_state_dict(weights_unpack_variant_semantic_or_none(semantic_blob, model.state_dict()), strict=True)
    carrier_blob, selector_blob = prediction_split_frame0_selector_carrier(parts.carrier_blob)
    carrier = prediction_materialize_cpr1(carrier_blob, render_namespace)
    scales, basis_codes, coefficient_scales, encoded = carrier_decode_compact_carrier(
        carrier, basis_count=12 * 3 * 24 * 32, frames=600, dimensions=12)
    codes = torch.from_numpy(encoded.astype(np.int32, copy=False))
    delta = (codes >> 1) ^ -(codes & 1)
    codes = torch.cumsum(delta, dim=0) & 0xFFF
    codes = torch.where(codes >= 0x800, codes - 0x1000, codes)
    basis = torch.from_numpy(basis_codes.astype(np.int8, copy=False)).reshape(12, 3, 24, 32).float()
    basis = basis * torch.from_numpy(scales)[:, None, None, None]
    coefficients = codes.reshape(600, 12).float() * torch.from_numpy(coefficient_scales)[None]
    return parts, model, basis, coefficients, selector_blob


def write_video(model, basis, coefficients, tokens, selector_blob, destination, device):
    render_render_video(model, basis, coefficients, tokens, destination, device)
    modes, indices = selector_decode_selector(selector_blob)
    raw = np.memmap(destination, mode='r+', dtype=np.uint8, shape=(1200, 874, 1164, 3))
    for mode_index, mode in enumerate(modes):
        frame_ids = np.flatnonzero(indices == mode_index)
        if frame_ids.size:
            raw[2 * frame_ids] = selector_apply_pixel_mode(np.asarray(raw[2 * frame_ids]).copy(), mode)
    raw.flush()


def main():
    if len(sys.argv) != 4:
        raise SystemExit('usage: inflate.sh <archive-dir> <output-dir> <file-list>')
    data_dir, output_dir, file_list = map(Path, sys.argv[1:])
    names = file_list.read_text().split()
    if names != ['0.mkv']:
        raise SystemExit('this archive contains only 0.mkv')
    archive = Path(__file__).with_name('archive.zip')
    content = archive.read_bytes()
    if len(content) != 179286 or hashlib.sha256(content).hexdigest() != 'aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957':
        raise SystemExit('archive.zip has an unexpected size or SHA-256')
    with archive_zipfile.ZipFile(archive) as compressed:
        if compressed.namelist() != ['p'] or (data_dir / 'p').read_bytes() != compressed.read('p'):
            raise SystemExit('extracted payload does not match archive.zip')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
    with torch.inference_mode():
        parts, model, basis, coefficients, selector = read_models(archive)
        tokens = torch.stack(list(TokenDecoder(parts, device)))
        write_video(model, basis, coefficients, tokens, selector, output_dir / '0.raw', device)


if __name__ == '__main__':
    main()
