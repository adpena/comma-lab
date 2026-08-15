# Borrowed-substrate accounting for e480b RX2

This accounting is deliberately mechanism-level. The exact archive is a new
composition and exact-byte result, but it is not a wholly original learned
vehicle. The categories are closed: `ours-original`,
`PR130-lineage-retrained-on-our-labels`, or `PR130/135-byte-identical`.

| Section or mechanism | Classification | Exact receipt | What the classification means |
|---|---|---|---|
| Semantic renderer physical state | `PR130/135-byte-identical` | decoded physical section: 36,040 B, SHA-256 `b0d41ec904aca82f93f3c8bc68d0e48896ba08efdaa7a4a2ee204f002fc28ec8`; shipped split-Brotli section: 34,763 B | The learned semantic renderer comes from the PR130/PR135 lineage and the retained reproduce has the same bytes. No new semantic-renderer originality is claimed. |
| Carrier physical state | `PR130/135-byte-identical` | decoded physical section: 22,219 B, SHA-256 `065fce08fc3d44e49d29ad624561cbef86d01282cc73dcd32533b5d63115bd9f`; shipped split-Brotli section: 22,161 B | The learned carrier comes from the PR130/PR135 lineage and the retained reproduce has the same bytes. |
| HPAC IHS1 probability object | `PR130-lineage-retrained-on-our-labels` | decoded IHS1: 17,996 B, SHA-256 `94526d667a9c8b98f1e3ef8d39fe8769d6cc6721cb9a102629ad47f26016460d`; shipped q10 section: 13,619 B | The architecture descends from PR130, but this object was trained on the exact current MC36 label field rather than copied from the ancestor. |
| RC64 token stream | `PR130-lineage-retrained-on-our-labels` | shipped stream: 112,749 B, SHA-256 `b981b8399f184795da7cd99b8ee44416bd672c8c4ed1672f1252b32a64c10627`; decoded 117,964,800-byte label field SHA-256 `9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52` | The entropy probabilities are from the retrained current-label object; decoded labels remain exactly the MC36 field. |
| Current-label correction table and retained residual | `ours-original` | table: 100 B, SHA-256 `3572a0db3d511f2c26b0ade0734e11112fec3f068bcba5900b54a0646eae61ec`; residual: 96 B, SHA-256 `64bbf9dfd88d6eb50d111f72d968ab7e8f8dc0ab00fb675d8ed2ee8a410b73ac` | The table was fitted for the deployed current-label checkpoint and the residual was carried through the exact receiver-identity race. |
| RX1M split-section container and lossless selection | `ours-original` | model wrapper: 70,557 B, SHA-256 `7cf390160189e8708faf3a7b09a76fc18cee85e45fdc7f71d30f725014417411`; layout = 14-byte header + HPAC q10 + semantic q11 + carrier q11 | The container, per-section coder selection, parse-back gates, and deterministic rebuild are repository work; the learned section contents retain the lineage classifications above. |
| F26 device-flexible CPU port and e480b receiver binding | `ours-original` | executable portable content-tree SHA-256 `26c7d4f6a8e111c071d74208fc625bf2358e077a06dc59b54ec9421a8d198e0b`; `inflate.py` SHA-256 `3da9fda50428ab6e2e9de28d3260335b0b04a92eca5632abd8ee042bcacb9928`; `runtime/f26_inflate.py` SHA-256 `2da706538755d55bade782f24558e1e61992f177c2e9cc9f06ab0d24f2574182` | This is the device-flexible adapter and new model-section reader that make the packet CPU-runnable. The underlying F26 renderer and learned vehicle remain borrowed lineage. |
| Final member and archive assembly | `ours-original` | member `p`: 183,402 B, SHA-256 `30c0165ec56dd9327ca4dcda477c34c25f7664622ac37ec8ed171114267d1b58`; archive: 183,502 B, SHA-256 `e3e6f440b45bbb92f2eeb58c7a56d74b3cd0a62bbcff01a26adcd008391c19d3` | The exact composition, receiver closure, deterministic repeat, and authority-row custody are new; they do not erase the section-level borrowing above. |

The result claim is therefore narrow: a new exact-byte composition and
current-label probability retraining on a PR130/PR135 learned substrate, with
an original CPU-capable receiver port and custody chain.
