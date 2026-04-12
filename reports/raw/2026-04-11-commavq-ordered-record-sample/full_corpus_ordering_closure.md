# Full Corpus Ordering Closure

- corpus size: `5000` records
- original bytes: `960000000`

## exact results

- canonical:
  - archive bytes: `462023463`
  - ratio: `2.0778165545242016`
  - artifact: `canonical_5000_tool.json`
- clip_greedy_nn:
  - archive bytes: `462023462`
  - ratio: `2.0778165590214117`
  - artifact: `clip_greedy_nn_5000_tool.json`
- pose_label_grouped_v2:
  - archive bytes: `462023462`
  - ratio: `2.0778165590214117`
  - artifact: `pose_label_grouped_5000_v2_tool.json`
- label_lexicographic_clip_rank:
  - archive bytes: `462023462`
  - ratio: `2.0778165590214117`
  - artifact: `label_lexicographic_clip_rank_5000_tool.json`

## conclusion

- all non-canonical orderers tied exactly
- each beat canonical by `1` byte
- the current ordering family is real but exhausted
- next gains must come from a different family or a materially stronger ranker/semantic labeler
