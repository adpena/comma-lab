# Compact video decoder

`archive.zip` holds a single 179,186-byte file: a small network that predicts a five-class label for every pixel, the
compressed field of those labels for 600 frame pairs, a second small network that turns labels into colour, and a
compact description of how the camera moves. Every learned number lives in the archive. `FORMAT.md` gives the layout,
byte by byte.

**How decoding works.** The decoder rebuilds both networks from the archive. It then reads the label field one frame
at a time: the prediction network proposes probabilities for a group of pixels, adaptive models sharpen them from the
pixels already read, and an arithmetic decoder pulls the group's labels out of the compressed stream. The colour
network draws the second frame of each pair from its labels, while the camera description and a short list of stored
pixel edits produce the first frame.

**What frame 0 of each pair is.** Only the second frame of a pair is a picture. The first is a smooth field centred on
mid-grey: twelve coefficients for that pair weight twelve stored three-channel 24 x 32 basis images, and the sum is
scaled to `127.5 + 64 x carrier` and enlarged — first to 384 x 512, then to the full 874 x 1,164 — with no scene
content in it at all. That is deliberate. The scoring pipeline reads each pair as a *pair*: one of its two measurements
compares the camera motion between the two frames, and the other looks only at the second. So the first frame's whole
job is to carry the motion signal, and twelve coefficients say it in about 31 bytes per frame — 18,483 bytes for all
600 of them, shared basis included — where a picture would cost far more. The seven whole-frame pixel edits — add or
subtract one from every channel, shift the frame one pixel, lay down a checkerboard of plus and minus one — are the
fine adjustment on top: a three-bit label, on 24 of the 600 pairs in this archive, nudges that field into the position
the motion measurement reads best. `PIXEL_MODES` in `inflate.py` lists all eight; label 0 means "leave the frame alone".

**How to run it.**

    python3 -m zipfile -e archive.zip extracted
    sh inflate.sh extracted inflated ../../public_test_video_names.txt

`inflate.sh` compiles the three C files with `cc`, then runs `inflate.py`. The output is `inflated/0.raw`: 1,200
frames of 874 x 1,164 RGB bytes, **3,662,409,600 bytes in one file, so the output directory needs about 3.5 GiB free**.
The decoder prints one JSON summary line at the end — how many pairs it read, the SHA-256 of the archive it read and
of the file it wrote, and the position the arithmetic decoder stopped at. Nothing else goes to stdout.

**What it needs.** Python 3.11 or newer, NumPy, PyTorch with CUDA, Brotli, and a C compiler named `cc`. NumPy and
PyTorch are assumed present: the evaluation environment ships them, so `inflate.py` imports them directly and a
missing one surfaces as Python's own `ImportError` naming the module. Brotli is checked and named explicitly, because
it is the one dependency that may not be there. There is no second implementation: if `cc` is missing,
`inflate.sh` says so in one line and stops; if a C file fails to compile, the compiler's own error is the last thing
printed and nothing is decoded.

**The list of videos.** The decoder stops unless the file list it is given is exactly `0.mkv`. This archive holds that
one video and nothing else, so any other list asks for frames that are not here; stopping beats writing the wrong ones.

**What has been measured.** On an NVIDIA Tesla T4 with CUDA, decoding this archive took 1,045.8 seconds and scoring
the frames took 44.3 seconds, giving 0.1361014714463198 over all 600 pairs. That run used this archive and this
decoding path. What has changed since is comments, names, tighter header checks and clearer error messages — nothing
that alters what is decoded — and the decoder is checked against that run to write byte-identical output.

The label field is decoded with integer arithmetic, so it is the same on every machine. The colour drawing uses
floating-point convolutions, so frames repeat exactly on the same device but can differ between a GPU and a CPU.
CUDA is the supported device; `--device cpu` exists for debugging and says so when it is used.

**Why three C files ship.** `corrector.c`, `geometry.c` and `range_decoder.c` hold the three loops whose order of
operations must not drift — fixed-order float64 probability mixing, exact integer geometry, and the arithmetic decoder
— and C states that order plainly and runs it fast enough for the time budget. Archive SHA-256
`aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957`, 179,286 bytes.
