import os
from multiprocessing import cpu_count

from nussl import AudioSignal
import os
import glob
from multiprocessing import cpu_count
from argparse import ArgumentParser

import shutil
import logging
import p_tqdm

def split_audio_file(original_path, split_path, sample_rate, verbose=False):
    """
    Splits an audio file at one path and places it at another path at a specified
    sample rate.
    
    Args:
        original_path (str): Path of audio file to be resampled.
        split_path (str): Path to save split audio files to.
        sample_rate (int): Sample rate to resample audio file to.
    """
    audio_signal = AudioSignal(original_path)
    resample = True

    if os.path.exists(resample_path):
        resampled_signal = AudioSignal(resample_path)
        resample = resampled_signal.sample_rate != sample_rate
    
    if resample:
        if verbose:
            logging.info(
                f'{original_path} @ {audio_signal.sample_rate} -> {resample_path} @ {sample_rate}'
            )
        audio_signal.resample(sample_rate)
        audio_signal.write_audio_to_file(resample_path)

def ig_f(dir, files):
    """
    Filter for making sure something is a file.
    
    Args:
        dir (str): Directory to filter to only look for files.
        files (list): List of items to filter.
    
    Returns:
        list: Filtered list.
    """
    return [f for f in files if os.path.isfile(os.path.join(dir, f))]

def resample(input_path, output_path, sample_rate, num_workers=1, 
             audio_extensions=['.wav', '.mp3', '.aac']):
    """
    Resamples a folder of audio files into a copy of the same folder with the same 
    structure but with every audio file replaced with a resampled version of that
    audio file. Relative paths to the audio file from the root of the folder will be the
    same.
    
    Args:
        input_path (str): Root of folder where all audio files will be resampled.
        output_path (str): Root of folder where all resampled files will be placed. Will match
            the same structure as the input_path folder structure.
        sample_rate (int): Sample rate to resample files to.
        num_workers (int, optional): How many workers to use in parallel to resample files. 
            Defaults to 1.
        audio_extensions (list, optional): Audio extensions to look for in the input_path. 
            Matching ones will be resampled and placed in the output_path at the 
            same relative location. Defaults to ['.wav', '.mp3', '.aac'].
    """
    try:
        shutil.copytree(input_path, output_path, ignore=ig_f)
    except:
        pass

    input_audio_files = []
    for ext in audio_extensions:
        input_audio_files += glob.glob(
            f"{input_path}/**/*{ext}", 
            recursive=True
        )

    output_audio_files = [
        x.replace(input_path, output_path)
        for x in input_audio_files
    ]

    indices = list(range(len(input_audio_files)))

    args = [
        [input_audio_files[i] for i in indices],
        [output_audio_files[i][:-4] + '.wav' for i in indices],
        sample_rate,
        False
    ]

    p_tqdm.p_map(resample_audio_file, *args, num_cpus=num_workers)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_path', type=str, 
        help="""Root of folder where all audio files will be resampled."""
    )
    parser.add_argument(
        '--output_path', type=str, 
        help="""Root of folder where all resampled files will be placed. Will match
        the same structure as the input_path folder structure."""
    )
    parser.add_argument(
        '--sample_rate', type=int, 
        help="""Sample rate to resample files to."""
    )
    parser.add_argument(
        '--num_workers', type=int, 
        help="""How many workers to use in parallel to resample files.""",
        default=1
    )
    parser.add_argument(
        '--audio_extensions', nargs='+', 
        help="""Audio extensions to look for in the input_path. Matching ones will
        be resampled and placed in the output_path at the same relative location.""",
        default=['.wav', '.mp3', '.aac']
    )
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    resample(args.input_path, args.output_path, args.sample_rate, args.num_workers,
             args.audio_extensions)