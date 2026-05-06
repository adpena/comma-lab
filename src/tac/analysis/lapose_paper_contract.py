"""Paper-alignment metadata for LA-Pose-inspired planning tools."""

from __future__ import annotations

LAPOSE_PAPER_REFERENCE = {
    "title": "LA-Pose: Latent Action Pretraining Meets Pose Estimation",
    "arxiv": "2604.27448",
    "project_page": "https://la-pose.github.io/",
    "implementation_alignment": "inspired_planning_only_not_paper_faithful_model",
    "missing_paper_components": [
        "inverse_forward_dynamics_pretraining",
        "video_tokenizer",
        "latent_action_encoder",
        "pose_head_translation_quaternion_fov_metric_scale",
        "waymo_nuscenes_argoverse_post_training",
    ],
}

__all__ = ["LAPOSE_PAPER_REFERENCE"]
