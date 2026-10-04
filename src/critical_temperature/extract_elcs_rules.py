"""Export fitted final eLCS rules; inspect frozen exports without refitting."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()


def export_rules(model, config, features, raw_train, selected_train, destination):
    """Export the exact fitted population before holdout evaluation.

    Upstream Accuracy is training rule accuracy, not validation accuracy. Row IDs
    are original dataset positions, with full-row deduplication applied only to
    the training partition. No fitting or scoring occurs here.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Rules already exist: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_ids = [int(row) for row in raw_train]
    selected_ids = [int(row) for row in selected_train]
    ensemble = config['name'] in ('ensemble', 'ensemble_dedup')
    seeds = config['parameters'].get('seeds', [])
    members = model.members if ensemble else [model]
    if ensemble and len(members) != len(seeds):
        raise ValueError('Fitted ensemble does not match frozen seeds.')
    metadata = {
        'model': config, 'model_sha256': json_hash(config),
        'feature_names': list(features), 'class_name': 'above_77k',
        'training_rows_raw': len(raw_ids), 'training_rows': len(selected_ids),
        'training_row_ids_raw': raw_ids, 'training_row_ids': selected_ids,
        'training_row_ids_raw_sha256': json_hash(raw_ids),
        'training_row_ids_sha256': json_hash(selected_ids),
        'accuracy_semantics': 'Upstream training rule Accuracy; not validation accuracy.',
        'members': [],
    }
    # Publish only a complete export; upstream failures leave no rules directory.
    with tempfile.TemporaryDirectory(prefix='.rules-', dir=destination.parent) as tmp:
        stage = Path(tmp) / 'rules'
        stage.mkdir()
        for index, member in enumerate(members):
            relative = f'member_{seeds[index]}' if ensemble else '.'
            folder = stage / relative
            folder.mkdir(exist_ok=True)
            population = folder / 'rules.csv'
            member.export_final_rule_population(
                headerNames=np.asarray(features), className='above_77k',
                filename=str(population), DCAL=True)
            frame = pd.read_csv(population)
            frame.sort_values(['Accuracy', 'Match Count', 'Fitness'], ascending=False,
                              kind='stable').head(20).to_csv(folder / 'top_rules.csv', index=False)
            parameters = {k: v for k, v in config['parameters'].items() if k != 'seeds'}
            if ensemble:
                parameters['random_state'] = seeds[index]
            metadata['members'].append({'directory': relative, 'parameters': parameters})
        metadata['files'] = {path.relative_to(stage).as_posix(): file_hash(path)
                             for path in sorted(stage.rglob('*.csv'))}
        (stage / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        stage.rename(destination)


def inspect_experiment(experiment, *, quiet=False):
    """Validate completed final artifacts and report already-saved rule paths."""
    output = Path(experiment)
    completion = output / 'test_complete.json'
    if not completion.is_file():
        raise ValueError('No completed final test export. Run the frozen final test explicitly first; no refit is performed here.')
    manifest = json.loads(completion.read_text(encoding='utf-8'))
    required = {'config.json', 'test_started.json', 'test_results.csv',
                'test_predictions.csv', 'rules/metadata.json'}
    if not required.issubset(manifest):
        raise ValueError('Final completion manifest does not include rule exports.')
    for name, digest in manifest.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid final artifact path.')
        path = output / relative
        if not path.is_file() or file_hash(path) != digest:
            raise ValueError(f'Final artifact changed or missing: {name}')
    config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    started = json.loads((output / 'test_started.json').read_text(encoding='utf-8'))
    meta = json.loads((output / 'rules' / 'metadata.json').read_text(encoding='utf-8'))
    if config != started or meta['model'] != config['model'] or meta['model_sha256'] != json_hash(config['model']):
        raise ValueError('Rules do not match the frozen final configuration.')
    results = pd.read_csv(output / 'test_results.csv')
    if len(results) != 1 or results.iloc[0]['fold'] != 'test':
        raise ValueError('Expected one final test result.')
    for field in ('training_rows_raw', 'training_rows'):
        if results.iloc[0][field] != meta[field]:
            raise ValueError('Rules do not match final training counts.')
    model = config['model']
    directories = ([f'member_{seed}' for seed in model['parameters']['seeds']]
                   if model['name'] in ('ensemble', 'ensemble_dedup') else ['.'])
    expected = {str(Path(directory) / name).replace('\\', '/')
                for directory in directories for name in ('rules.csv', 'top_rules.csv')}
    if set(meta['files']) != expected or [member['directory'] for member in meta['members']] != directories:
        raise ValueError('Rules are missing a required fitted member or population file.')
    for name, digest in meta['files'].items():
        if manifest.get(f'rules/{name}') != digest:
            raise ValueError('Rule files are missing from the final manifest.')
        if not quiet:
            print(output / 'rules' / name)
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True,
                        help='Completed frozen experiment to validate and inspect; never refits.')
    args = parser.parse_args(argv)
    inspect_experiment(args.experiment)


if __name__ == '__main__':
    main()
