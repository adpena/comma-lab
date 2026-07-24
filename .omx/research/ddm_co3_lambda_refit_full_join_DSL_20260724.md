# DDM CO3 N600 lambda refit — typed control DSL

```text
organ ddm_co3_lambda_ranker {
  maturity              = "_dev"
  research_only         = true
  execution_allowed     = false
  actuation             = NONE
  score_claim           = false
  promotion_eligible    = false
  main_review_required  = true

  population {
    pair_ids       = 0..599
    heldout_unit   = source_pair_id
    outer_folds    = sha256_mod_5("ddm-co3-n600-v1:")
    report_surface = concatenated_out_of_fold_only
  }

  sources {
    g3_atlas          = HASH_REQUIRED
    ev1_receiver_join = HASH_REQUIRED
    ms4d_margin       = ORACLE(FRESH, bucket_rows=1200, direct_blocks=25)
    ms4d_pose_tube    = ORACLE(FRESH, pair_rows=600)
    g4_stationarity   = ORACLE(FRESH, strata=5, classes=3)
    rd1_duals         = HASH_REQUIRED
    preregistration   = HASH_REQUIRED
  }

  race {
    factorized_refit
    factorized_ms4d_interactions
    g4_regime_conditional
    small_monotone_gb IF all(first_three.ndcg_at_4 < 0.75)
    close_form_mixture IF inner_ndcg_gap <= 0.02
  }

  select {
    primary   = heldout_ndcg_at_4
    tiebreak  = heldout_spearman
    final_tie = lower_complexity
    admit_if  = heldout_ndcg_at_4 >= 0.75
  }

  decide {
    IF admitted AND fisher_precision_complete {
      pair_duty_ranking = ADVISORY_RANK_ELIGIBLE
    } ELSE IF admitted {
      emit_duty CO3_LAMBDA_RANKER_FISHER_PRECISION_CLOSURE
      pair_duty_ranking = BLOCKED_INCOMPLETE_FISHER_PRECISION
    } ELSE {
      emit_blocker BLOCKED_CO3_HELDOUT_NDCG_ADMISSION
    }

    require_blocker BLOCKED_J8F_REALIZED_VERDICT_TELEMETRY
    never_actuate
  }

  consumers = [digest, dashboard, duty_queue, activation_nag]
  invariant all(consumers.state_digest == campaign.state_digest)
}
```

The DSL is descriptive of the implemented typed contract. It introduces no
trainer flag, launcher, provider path, or executable actuation surface.
