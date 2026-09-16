import argparse
import hashlib
import sys
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('file_list', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.file_list.read_text().split() != ['0.mkv']:
        raise SystemExit('this archive contains only 0.mkv')
    archive = Path(__file__).with_name('archive.zip')
    content = archive.read_bytes()
    if len(content) != 179286 or hashlib.sha256(content).hexdigest() != 'aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957':
        raise SystemExit('archive.zip has an unexpected size or SHA-256')
    with zipfile.ZipFile(archive) as compressed:
        if compressed.namelist() != ['p'] or (args.archive_dir / 'p').read_bytes() != compressed.read('p'):
            raise SystemExit('extracted payload does not match archive.zip')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is required by default; --device cpu explicitly selects CPU output, which may differ.')
    if args.device == 'cpu':
        print('CPU selected explicitly; output may differ from CUDA.', file=sys.stderr)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    if device.type == 'cuda':
        try:
            torch.backends.cuda.matmul.fp32_precision = 'ieee'
            torch.backends.cudnn.conv.fp32_precision = 'ieee'
        except (AttributeError, RuntimeError):
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        parts, model, basis, coefficients, selector = read_models(archive)
        tokens = torch.stack(list(TokenDecoder(parts, device)))
        write_video(model, basis, coefficients, tokens, selector, args.output_dir / '0.raw', device)


if __name__ == '__main__':
    main()
