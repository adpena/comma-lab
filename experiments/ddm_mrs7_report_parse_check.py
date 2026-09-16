"""Standalone check that the frozen contract accepts submissions/mrs7's JSON summary line.

Reads the cold decode's own stdout log from the identity run and hands it to
`tac.decode_wall_clock._cold_public_report` and `tac.candidate_seal._pf_cold_work_facts`
exactly as the evaluation harness would. Prints the facts both readers extract.

Axis: [macOS-CPU advisory]. score_claim=false, promotable=false. No score is claimed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stdout-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    arguments = parser.parse_args()

    from tac.candidate_seal import _pf_cold_work_facts
    from tac.decode_wall_clock import _cold_public_report

    text = arguments.stdout_log.read_text(encoding='utf-8')
    # The harness never sees a clean line: it scans a whole log. Surround it with noise so this
    # check exercises the same search the harness does.
    noisy = 'preamble that is not json\n' + text + '\ntrailing noise {not json}\n'
    report = _cold_public_report({'artifacts': {'contest_auth_eval.stdout.log': noisy}})
    facts = _pf_cold_work_facts(report, 'ddm_mrs7_report_parse_check')
    value = dict(stdout_log=str(arguments.stdout_log), cold_public_report=report,
                 cold_work_facts=facts, accepted=True,
                 axis='[macOS-CPU advisory]', score_claim=False, promotable=False)
    arguments.output.write_text(json.dumps(value, indent=2, sort_keys=True))
    print(json.dumps(dict(accepted=True, pair_count=report['pair_count'],
                          checkpoint_resume=report['checkpoint_resume'],
                          token_cache=report['token_cache'], facts=facts), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
