//! Native Rust/Rayon combo search for DQS1 pairset component marginals.
//!
//! This crate is a speed layer only. The Python planner owns semantic custody
//! and false-authority enforcement; this binary ranks candidate pair
//! combinations from already-canonicalized marginal rows.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashSet};

const FALSE: bool = false;

#[derive(Debug, Deserialize)]
pub struct PlannerRequest {
    pub schema: Option<String>,
    #[serde(default = "default_beam_width")]
    pub beam_width: usize,
    #[serde(default = "default_pool_limit")]
    pub pool_limit: usize,
    #[serde(default)]
    pub groups: Vec<ComboGroup>,
}

#[derive(Debug, Deserialize)]
pub struct ComboGroup {
    pub group_id: String,
    pub base_pair_indices: Vec<u16>,
    #[serde(default)]
    pub combo_counts: Vec<usize>,
    #[serde(default)]
    pub existing_pairsets: Vec<Vec<u16>>,
    #[serde(default)]
    pub existing_candidate_ids: Vec<String>,
    #[serde(default)]
    pub rows: Vec<ComboRow>,
    #[serde(default)]
    pub pairwise_interactions: Vec<PairwiseInteraction>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ComponentDeltas {
    pub segnet_delta: f64,
    pub posenet_delta: f64,
    pub rate_delta: f64,
}

impl ComponentDeltas {
    fn zero() -> Self {
        Self {
            segnet_delta: 0.0,
            posenet_delta: 0.0,
            rate_delta: 0.0,
        }
    }

    fn add_assign(&mut self, other: &Self) {
        self.segnet_delta += other.segnet_delta;
        self.posenet_delta += other.posenet_delta;
        self.rate_delta += other.rate_delta;
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ComboRow {
    pub candidate_id: String,
    pub pair_index: u16,
    pub dropped_pair_rank: u16,
    pub net_component_delta: f64,
    pub component_deltas: ComponentDeltas,
    pub axis_baseline_score: Option<f64>,
    pub component_marginal_status: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PairwiseInteraction {
    pub candidate_id: Option<String>,
    pub pair_indices: Vec<u16>,
    pub net_interaction_delta: f64,
    pub interaction_component_deltas: ComponentDeltas,
    pub interaction_status: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PlannerResponse {
    pub schema: &'static str,
    pub planner: &'static str,
    pub search_strategy: &'static str,
    pub group_count: usize,
    pub combo_count: usize,
    pub score_claim: bool,
    pub score_claim_valid: bool,
    pub promotion_eligible: bool,
    pub rank_or_kill_eligible: bool,
    pub ready_for_exact_eval_dispatch: bool,
    pub promotable: bool,
    pub dispatch_attempted: bool,
    pub gpu_launched: bool,
    pub combos: Vec<ComboSpec>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PairwiseInteractionTerm {
    pub source_candidate_id: Option<String>,
    pub pair_indices: Vec<u16>,
    pub net_interaction_delta: f64,
    pub interaction_status: Option<String>,
    pub score_claim: bool,
    pub score_claim_valid: bool,
    pub promotion_eligible: bool,
    pub rank_or_kill_eligible: bool,
    pub ready_for_exact_eval_dispatch: bool,
    pub promotable: bool,
    pub dispatch_attempted: bool,
    pub gpu_launched: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ComboScore {
    pub first_order_net_component_delta: f64,
    pub second_order_net_component_delta: f64,
    pub expected_net_component_delta: f64,
    pub expected_component_deltas: ComponentDeltas,
    pub pairwise_interaction_terms: Vec<PairwiseInteractionTerm>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ComboSpec {
    pub group_id: String,
    pub candidate_id: String,
    pub base_pair_indices: Vec<u16>,
    pub dropped_pair_indices: Vec<u16>,
    pub dropped_pair_ranks: Vec<u16>,
    pub selected_pair_indices: Vec<u16>,
    pub score: ComboScore,
    pub source_rows: Vec<ComboRow>,
    pub search_strategy: &'static str,
    pub rank_in_group: usize,
}

#[derive(Clone, Debug)]
struct BeamState {
    indices: Vec<usize>,
    last_index: isize,
    dropped_pair_indices: Vec<u16>,
    dropped_pair_ranks: Vec<u16>,
    score: ComboScore,
}

fn default_beam_width() -> usize {
    64
}

fn default_pool_limit() -> usize {
    48
}

pub fn plan_combos(request: &PlannerRequest) -> Result<PlannerResponse, String> {
    if let Some(schema) = &request.schema {
        if schema != "pairset_component_combo_request.v1" {
            return Err(format!("unsupported schema: {schema}"));
        }
    }
    if request.beam_width == 0 {
        return Err("beam_width must be positive".to_string());
    }
    if request.pool_limit == 0 {
        return Err("pool_limit must be positive".to_string());
    }

    let mut combos: Vec<ComboSpec> = request
        .groups
        .par_iter()
        .flat_map(|group| plan_group(group, request.beam_width, request.pool_limit))
        .collect();
    combos.sort_by(combo_sort);
    for (index, combo) in combos.iter_mut().enumerate() {
        combo.rank_in_group = index + 1;
    }
    Ok(PlannerResponse {
        schema: "pairset_component_combo_response.v1",
        planner: "pairset-combo-planner",
        search_strategy: "rust_rayon_beam_pairwise_interaction_waterfill",
        group_count: request.groups.len(),
        combo_count: combos.len(),
        score_claim: FALSE,
        score_claim_valid: FALSE,
        promotion_eligible: FALSE,
        rank_or_kill_eligible: FALSE,
        ready_for_exact_eval_dispatch: FALSE,
        promotable: FALSE,
        dispatch_attempted: FALSE,
        gpu_launched: FALSE,
        combos,
    })
}

fn plan_group(group: &ComboGroup, beam_width: usize, pool_limit: usize) -> Vec<ComboSpec> {
    let mut rows: Vec<ComboRow> = group
        .rows
        .iter()
        .filter(|row| row.net_component_delta < 0.0)
        .cloned()
        .collect();
    rows.sort_by(row_sort);
    rows.truncate(pool_limit);
    if rows.len() < 2 {
        return Vec::new();
    }

    let requested_counts: BTreeSet<usize> = group.combo_counts.iter().copied().collect();
    let Some(max_count) = requested_counts.iter().max().copied() else {
        return Vec::new();
    };
    let max_count = max_count.min(rows.len());
    let existing_pairsets: HashSet<Vec<u16>> = group
        .existing_pairsets
        .iter()
        .map(|pairs| sorted_unique(pairs))
        .collect();
    let existing_ids: HashSet<&str> = group
        .existing_candidate_ids
        .iter()
        .map(String::as_str)
        .collect();
    let base_pairs = sorted_unique(&group.base_pair_indices);
    let mut emitted_pairsets: HashSet<Vec<u16>> = HashSet::new();
    let mut emitted_ids: HashSet<String> = HashSet::new();
    let mut states = vec![BeamState {
        indices: Vec::new(),
        last_index: -1,
        dropped_pair_indices: Vec::new(),
        dropped_pair_ranks: Vec::new(),
        score: score_rows(&[], &group.pairwise_interactions),
    }];
    let mut out = Vec::new();
    for count in 1..=max_count {
        let mut expanded = Vec::new();
        for state in &states {
            let used_pairs: HashSet<u16> = state.dropped_pair_indices.iter().copied().collect();
            for index in (state.last_index + 1) as usize..rows.len() {
                let row = &rows[index];
                if used_pairs.contains(&row.pair_index) {
                    continue;
                }
                let mut indices = state.indices.clone();
                indices.push(index);
                let source_rows: Vec<&ComboRow> = indices.iter().map(|i| &rows[*i]).collect();
                let score = score_rows(&source_rows, &group.pairwise_interactions);
                let mut dropped_pairs: Vec<u16> =
                    source_rows.iter().map(|row| row.pair_index).collect();
                dropped_pairs.sort_unstable();
                dropped_pairs.dedup();
                let mut ranks: Vec<u16> = source_rows
                    .iter()
                    .map(|row| row.dropped_pair_rank)
                    .collect();
                ranks.sort_unstable();
                expanded.push(BeamState {
                    indices,
                    last_index: index as isize,
                    dropped_pair_indices: dropped_pairs,
                    dropped_pair_ranks: ranks,
                    score,
                });
            }
        }
        expanded.sort_by(state_sort);
        expanded.truncate(beam_width);
        states = expanded;
        if !requested_counts.contains(&count) {
            continue;
        }
        for state in &states {
            let selected_pairs = selected_pairs(&base_pairs, &state.dropped_pair_indices);
            if selected_pairs.is_empty()
                || existing_pairsets.contains(&selected_pairs)
                || emitted_pairsets.contains(&selected_pairs)
            {
                continue;
            }
            let candidate_id = combo_candidate_id(&state.dropped_pair_indices);
            if existing_ids.contains(candidate_id.as_str()) || emitted_ids.contains(&candidate_id) {
                continue;
            }
            let source_rows: Vec<ComboRow> =
                state.indices.iter().map(|i| rows[*i].clone()).collect();
            if !source_rows
                .iter()
                .any(|row| row.axis_baseline_score.is_some())
            {
                continue;
            }
            emitted_pairsets.insert(selected_pairs.clone());
            emitted_ids.insert(candidate_id.clone());
            out.push(ComboSpec {
                group_id: group.group_id.clone(),
                candidate_id,
                base_pair_indices: base_pairs.clone(),
                dropped_pair_indices: state.dropped_pair_indices.clone(),
                dropped_pair_ranks: state.dropped_pair_ranks.clone(),
                selected_pair_indices: selected_pairs,
                score: state.score.clone(),
                source_rows,
                search_strategy: "rust_rayon_beam_pairwise_interaction_waterfill",
                rank_in_group: out.len() + 1,
            });
            break;
        }
    }
    out
}

fn score_rows(rows: &[&ComboRow], interactions: &[PairwiseInteraction]) -> ComboScore {
    let mut components = ComponentDeltas::zero();
    let mut first_order = 0.0;
    let mut dropped_pairs = Vec::new();
    for row in rows {
        first_order += row.net_component_delta;
        components.add_assign(&row.component_deltas);
        dropped_pairs.push(row.pair_index);
    }
    dropped_pairs.sort_unstable();
    dropped_pairs.dedup();

    let mut second_order = 0.0;
    let mut terms = Vec::new();
    for (left_index, left) in dropped_pairs.iter().enumerate() {
        for right in dropped_pairs.iter().skip(left_index + 1) {
            if let Some(interaction) = find_interaction(interactions, *left, *right) {
                second_order += interaction.net_interaction_delta;
                components.add_assign(&interaction.interaction_component_deltas);
                terms.push(PairwiseInteractionTerm {
                    source_candidate_id: interaction.candidate_id.clone(),
                    pair_indices: vec![*left, *right],
                    net_interaction_delta: interaction.net_interaction_delta,
                    interaction_status: interaction.interaction_status.clone(),
                    score_claim: FALSE,
                    score_claim_valid: FALSE,
                    promotion_eligible: FALSE,
                    rank_or_kill_eligible: FALSE,
                    ready_for_exact_eval_dispatch: FALSE,
                    promotable: FALSE,
                    dispatch_attempted: FALSE,
                    gpu_launched: FALSE,
                });
            }
        }
    }
    ComboScore {
        first_order_net_component_delta: first_order,
        second_order_net_component_delta: second_order,
        expected_net_component_delta: first_order + second_order,
        expected_component_deltas: components,
        pairwise_interaction_terms: terms,
    }
}

fn find_interaction(
    interactions: &[PairwiseInteraction],
    left: u16,
    right: u16,
) -> Option<&PairwiseInteraction> {
    let key = ordered_pair(left, right);
    interactions.iter().find(|interaction| {
        if interaction.pair_indices.len() != 2 {
            return false;
        }
        ordered_pair(interaction.pair_indices[0], interaction.pair_indices[1]) == key
    })
}

fn selected_pairs(base_pairs: &[u16], dropped_pairs: &[u16]) -> Vec<u16> {
    let dropped: HashSet<u16> = dropped_pairs.iter().copied().collect();
    base_pairs
        .iter()
        .copied()
        .filter(|pair| !dropped.contains(pair))
        .collect()
}

fn sorted_unique(values: &[u16]) -> Vec<u16> {
    let mut out = values.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn ordered_pair(left: u16, right: u16) -> (u16, u16) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn combo_candidate_id(dropped_pairs: &[u16]) -> String {
    let pairs = sorted_unique(dropped_pairs);
    let mut suffix = pairs
        .iter()
        .take(6)
        .map(|pair| format!("p{pair:04}"))
        .collect::<Vec<_>>()
        .join("_");
    if pairs.len() > 6 {
        suffix.push_str(&format!("_plus{:02}", pairs.len() - 6));
    }
    format!("pairset_learned_drop_combo_k{:03}_{suffix}", pairs.len())
}

fn row_sort(left: &ComboRow, right: &ComboRow) -> Ordering {
    cmp_f64(left.net_component_delta, right.net_component_delta)
        .then(left.dropped_pair_rank.cmp(&right.dropped_pair_rank))
        .then(left.pair_index.cmp(&right.pair_index))
        .then(left.candidate_id.cmp(&right.candidate_id))
}

fn state_sort(left: &BeamState, right: &BeamState) -> Ordering {
    cmp_f64(
        left.score.expected_net_component_delta,
        right.score.expected_net_component_delta,
    )
    .then(left.dropped_pair_ranks.cmp(&right.dropped_pair_ranks))
    .then(left.dropped_pair_indices.cmp(&right.dropped_pair_indices))
}

fn combo_sort(left: &ComboSpec, right: &ComboSpec) -> Ordering {
    left.group_id
        .cmp(&right.group_id)
        .then(cmp_f64(
            left.score.expected_net_component_delta,
            right.score.expected_net_component_delta,
        ))
        .then(left.dropped_pair_ranks.cmp(&right.dropped_pair_ranks))
        .then(left.dropped_pair_indices.cmp(&right.dropped_pair_indices))
        .then(left.candidate_id.cmp(&right.candidate_id))
}

fn cmp_f64(left: f64, right: f64) -> Ordering {
    left.total_cmp(&right)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pair: u16, rank: u16, delta: f64) -> ComboRow {
        ComboRow {
            candidate_id: format!("drop_one_{pair}"),
            pair_index: pair,
            dropped_pair_rank: rank,
            net_component_delta: delta,
            component_deltas: ComponentDeltas {
                segnet_delta: 0.0,
                posenet_delta: 0.0,
                rate_delta: delta,
            },
            axis_baseline_score: Some(0.19203),
            component_marginal_status: Some("rate_credit_exceeds_scorer_penalty".to_string()),
        }
    }

    #[test]
    fn ranks_first_order_combos_when_no_interactions_exist() {
        let request = PlannerRequest {
            schema: Some("pairset_component_combo_request.v1".to_string()),
            beam_width: 8,
            pool_limit: 8,
            groups: vec![ComboGroup {
                group_id: "g".to_string(),
                base_pair_indices: vec![101, 327, 371, 376],
                combo_counts: vec![2],
                existing_pairsets: vec![],
                existing_candidate_ids: vec![],
                rows: vec![row(327, 2, -3.0), row(371, 3, -2.0), row(376, 4, -1.0)],
                pairwise_interactions: vec![],
            }],
        };
        let response = plan_combos(&request).expect("plan");
        assert_eq!(response.combos.len(), 1);
        assert_eq!(response.combos[0].dropped_pair_indices, vec![327, 371]);
        assert_eq!(response.combos[0].selected_pair_indices, vec![101, 376]);
        assert!(!response.score_claim);
    }

    #[test]
    fn avoids_pairwise_antagonism_when_second_order_term_dominates() {
        let request = PlannerRequest {
            schema: Some("pairset_component_combo_request.v1".to_string()),
            beam_width: 8,
            pool_limit: 8,
            groups: vec![ComboGroup {
                group_id: "g".to_string(),
                base_pair_indices: vec![101, 327, 371, 376],
                combo_counts: vec![2],
                existing_pairsets: vec![],
                existing_candidate_ids: vec![],
                rows: vec![row(327, 2, -3.0), row(371, 3, -2.0), row(376, 4, -1.0)],
                pairwise_interactions: vec![PairwiseInteraction {
                    candidate_id: Some("drop_two_bad".to_string()),
                    pair_indices: vec![327, 371],
                    net_interaction_delta: 10.0,
                    interaction_component_deltas: ComponentDeltas {
                        segnet_delta: 10.0,
                        posenet_delta: 0.0,
                        rate_delta: 0.0,
                    },
                    interaction_status: Some(
                        "scorer_penalty_expected_to_exceed_rate_credit".to_string(),
                    ),
                }],
            }],
        };
        let response = plan_combos(&request).expect("plan");
        assert_eq!(response.combos.len(), 1);
        assert_eq!(response.combos[0].dropped_pair_indices, vec![327, 376]);
        assert_eq!(
            response.combos[0].score.second_order_net_component_delta,
            0.0
        );
    }

    #[test]
    fn skips_existing_pairsets_and_candidate_ids() {
        let request = PlannerRequest {
            schema: Some("pairset_component_combo_request.v1".to_string()),
            beam_width: 8,
            pool_limit: 8,
            groups: vec![ComboGroup {
                group_id: "g".to_string(),
                base_pair_indices: vec![101, 327, 371, 376],
                combo_counts: vec![2],
                existing_pairsets: vec![vec![101, 376]],
                existing_candidate_ids: vec![],
                rows: vec![row(327, 2, -3.0), row(371, 3, -2.0), row(376, 4, -1.0)],
                pairwise_interactions: vec![],
            }],
        };
        let response = plan_combos(&request).expect("plan");
        assert_eq!(response.combos[0].dropped_pair_indices, vec![327, 376]);
        assert_eq!(response.combos[0].selected_pair_indices, vec![101, 371]);
    }
}
