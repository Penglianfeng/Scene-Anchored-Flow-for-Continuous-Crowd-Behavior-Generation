import os
import json
import numpy as np

from CrowdES.inference_model import CrowdESFramework
from utils.dataloader.evaluation_dataloader import EvaluationDataset
from utils.utils import reproducibility_settings
from utils.metrics import compute_metrics
from utils.visualization import produce_video_from_data


# Global settings
TRIALS = 20           # Number of trials for each scene. DO NOT CHANGE THIS!
EXPORT_VIDEO = False  # Export video for visualization
RUN_DIAGNOSTICS = False  # Run diagnostics for evaluation
MAX_EMPTY_GENERATION_RETRIES = 20


def _generate_nonempty_scenario(framework, scenario_length, primary_seed):
    """Generate a trial, deterministically resampling only empty outputs."""
    attempted_seeds = []
    for retry_idx in range(MAX_EMPTY_GENERATION_RETRIES + 1):
        generation_seed = primary_seed + retry_idx * TRIALS
        attempted_seeds.append(generation_seed)
        generated_scenario = framework.generate(scenario_length, seed=generation_seed)
        if not generated_scenario.empty:
            if retry_idx:
                print(
                    f'[empty-generation recovery] primary_seed={primary_seed}, '
                    f'used_seed={generation_seed}, retries={retry_idx}'
                )
            return generated_scenario, {
                'primary_seed': primary_seed,
                'used_seed': generation_seed,
                'retry_count': retry_idx,
                'attempted_seeds': attempted_seeds,
            }

        print(
            f'[empty-generation retry] seed={generation_seed} produced no valid '
            'trajectory after post-processing.'
        )

    raise RuntimeError(
        f'Generated scenario remained empty after {MAX_EMPTY_GENERATION_RETRIES} '
        f'retries; attempted seeds: {attempted_seeds}'
    )


def _save_generation_stats(path, records, total_trials):
    primary_empty_trials = sum(record['retry_count'] > 0 for record in records)
    completed_trials = len(records)
    payload = {
        'policy': {
            'name': 'deterministic_nonempty_resampling',
            'primary_seeds': list(range(TRIALS)),
            'fallback_seed_stride': TRIALS,
            'max_retries': MAX_EMPTY_GENERATION_RETRIES,
        },
        'completed_trials': completed_trials,
        'total_trials': total_trials,
        'primary_empty_trials': primary_empty_trials,
        'primary_empty_rate_completed': (
            primary_empty_trials / completed_trials if completed_trials else 0.0
        ),
        'total_empty_attempts': sum(record['retry_count'] for record in records),
        'records': records,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=4)


def main(config, seed=0):
    if RUN_DIAGNOSTICS:
        try:
            from utils.diagnostics import (
                compute_crowd_diagnostics,
                extract_core_diagnostic_scalars,
            )
        except ImportError as exc:
            raise ImportError(
                "RUN_DIAGNOSTICS requires the optional utils.diagnostics module."
            ) from exc

    # Reproducibility
    reproducibility_settings(seed=seed)

    # Load dataset and framework
    dataset_test = EvaluationDataset(config, 'test')
    framework = CrowdESFramework(config)

    # Start inference
    dataset_name = config.dataset.dataset_name
    emitter_checkpoint_dir = config.crowd_emitter.emitter.checkpoint_dir.format(dataset_name)
    run_tag = os.path.basename(os.path.normpath(emitter_checkpoint_dir)) or "emitter"
    scene_list = dataset_test.scene_list

    all_metrics = {}
    all_diagnostics_core = {}
    diagnostics_rows = []
    generation_records = []
    generation_stats_path = (
        f'./output/log/{dataset_name}/generation_stats_{dataset_name}_{TRIALS}_{run_tag}.json'
    )
    os.makedirs(os.path.dirname(generation_stats_path), exist_ok=True)
    for scene_idx, scene in enumerate(scene_list):
        for trial in range(TRIALS):
            print(f'Scene {scene_idx + 1}/{len(scene_list)}: {scene}, Trial {trial + 1}/{TRIALS}')

            # Load data
            data = dataset_test[scene_idx]
            scene_img = data['img']
            scene_seg = data['seg']
            scene_H = data['H']
            scene_walkable = data['walkable']
            scene_navmesh = data['navmesh']

            # Inference
            scenario_length = data['size']['length']
            framework.initialize_scene(scene_img, scene_seg, scene_walkable, scene_navmesh, scene_H)
            generated_scenario, generation_record = _generate_nonempty_scenario(
                framework,
                scenario_length,
                primary_seed=trial,
            )
            generation_record.update({'scene': scene, 'trial': trial})
            generation_records.append(generation_record)
            _save_generation_stats(
                generation_stats_path,
                generation_records,
                total_trials=len(scene_list) * TRIALS,
            )
            generated_scenario['scene'] = scene
            generated_dir = f'./output/generated/{dataset_name}/{run_tag}'
            os.makedirs(generated_dir, exist_ok=True)
            generated_scenario.to_csv(f'{generated_dir}/{scene}-{trial}.csv', index=False)

            # Evaluation
            # Print statistics
            gt_agents = data['size']['num_agents']
            print(f'GT scenario: {gt_agents} total agents')
            print(f"Generated scenario: {len(generated_scenario['agent_id'].unique())} total agents")

            # Compute metrics
            scene_size = data['size']
            scene_trajectory_dense = data['trajectory_dense']
            metrics = compute_metrics(generated_scenario, scene_trajectory_dense, scene_size, scene_H)
            print(f'Metrics: {metrics}')

            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = {}
                for k, v in value.items():
                    if k not in all_metrics[key]:
                        all_metrics[key][k] = []
                    all_metrics[key][k].append(v)

            if RUN_DIAGNOSTICS:
                diagnostics = compute_crowd_diagnostics(
                    source=generated_scenario,
                    target=scene_trajectory_dense,
                    fps=scene_size['fps'],
                    H=scene_H,
                    include_hit_counts=False,
                )
                diagnostics_core = extract_core_diagnostic_scalars(diagnostics)
                print(f'Diagnostics: {diagnostics_core}')
                diagnostics_rows.append({
                    'scene': scene,
                    'trial': trial,
                    'core': diagnostics_core,
                    'full': diagnostics,
                })
                for k, v in diagnostics_core.items():
                    if k not in all_diagnostics_core:
                        all_diagnostics_core[k] = []
                    all_diagnostics_core[k].append(v)
            
            # Export video for visualization
            if EXPORT_VIDEO:
                scene_bg = data['bg']
                video_path = f'./output/generated/{dataset_name}/{scene}-{trial}.avi'
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                produce_video_from_data(video_path, scene_img, scene_bg, generated_scenario, scenario_length, config)
                print(f'Video saved at {video_path}')

    # Print average metrics
    average_metrics = {}
    print('Average metrics:')
    for key, value in all_metrics.items():
        average_metrics[key] = {}
        for k, v in value.items():
            print(f'{key} {k}: {np.mean(v)}')
            average_metrics[key][k] = np.mean(v)

    # Save metrics to file
    metrics_path = f'./output/log/{dataset_name}/metrics_{dataset_name}_{TRIALS}_{run_tag}.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(average_metrics, f, indent=4)
    print(f'Metrics saved at {metrics_path}')

    if RUN_DIAGNOSTICS:
        average_diagnostics = {}
        std_diagnostics = {}
        print('Average diagnostics:')
        for k, v in all_diagnostics_core.items():
            average_diagnostics[k] = float(np.mean(v))
            std_diagnostics[k] = float(np.std(v))
            print(f'{k}: {average_diagnostics[k]} ± {std_diagnostics[k]}')

        diagnostics_path = f'./output/log/{dataset_name}/diagnostics_{dataset_name}_{TRIALS}_{run_tag}.json'
        with open(diagnostics_path, 'w') as f:
            json.dump(
                {
                    'mean': average_diagnostics,
                    'std': std_diagnostics,
                    'raw': diagnostics_rows,
                },
                f,
                indent=4,
            )
        print(f'Diagnostics saved at {diagnostics_path}')

    return average_metrics
