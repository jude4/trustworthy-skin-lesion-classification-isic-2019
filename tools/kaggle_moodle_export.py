"""Run this CELL content in your Kaggle notebook AFTER completing/saving the experiments.

Creates two ZIPs under /kaggle/working:
- ISIC2019_Kaggle_Code_Results.zip    (small; recommended for Moodle)
- ISIC2019_Kaggle_Model_Checkpoints.zip (large; upload only if allowed)

Export the complete executed .ipynb separately via Kaggle's notebook download menu.
The script deliberately does not include the ~9 GB raw image dataset.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

work = Path('/kaggle/working')

code_files = [
    'isic_fl_train.py',
    'isic2019_verified.csv',
    'isic2019_train.csv',
    'isic2019_val.csv',
    'isic2019_test.csv',
    'isic2019_federated_partitions.csv',
    'fl_history_1rounds.json',
    'fl_history_5rounds.json',
    'centralized_history.json',
    'model_comparison.csv',
    'per_class_comparison.csv',
    'centralized_classification_report.csv',
    'per_class_recall_comparison.csv',
    'centralized_training_loss.png',
    'centralized_validation_metrics.png',
    'centralized_confusion_matrix.png',
    'federated_validation_performance.png',
    'federated_confusion_matrix.png',
    'per_class_recall_comparison.png',
    'isic2019_samples.png',
]
model_files = ['resnet18_centralized_best.pt', 'fl_best_5rounds.pt']


def pack_zip(name, filenames, prefix):
    destination = work / name
    found, missing = [], []
    with ZipFile(destination, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for filename in filenames:
            src = work / filename
            if src.is_file():
                archive.write(src, f'{prefix}/{src.name}')
                found.append(filename)
            else:
                missing.append(filename)
    print(f'\nCreated: {destination} ({destination.stat().st_size / 1024**2:.1f} MB)')
    print('Included:', ', '.join(found) or '(none)')
    if missing:
        print('Not found (check whether you saved these outputs):', ', '.join(missing))
    return destination


small = pack_zip('ISIC2019_Kaggle_Code_Results.zip', code_files, 'kaggle_outputs')
large = pack_zip('ISIC2019_Kaggle_Model_Checkpoints.zip', model_files, 'models')
print('\nIMPORTANT: also download the FULL executed Kaggle notebook (.ipynb) from the Kaggle editor.')
print('Download both ZIPs from the Kaggle Output/Files panel. Check Moodle maximum file size.')
