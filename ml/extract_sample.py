"""
Narzędzie CLI do wycinania, segmentacji i masowego eksportowania próbek z dużych plików .mat.

Przykłady użycia:
    # 1. Rozbicie całego dużego pliku 1.3 GB (np. 80 prób) na folder pojedynczych małych plików:
    python -m ml.extract_sample --input D:/Dane/SEED-VII/SEED-VII/EEG_preprocessed/1.mat --split-all --output-dir data/samples_badany1/

    # 2. Wyciągnięcie pojedynczej próby (np. próba 5) do małego pliku:
    python -m ml.extract_sample --input EEG_preprocessed/1.mat --trial 5 --output data/sample_trial_5.mat

    # 3. Wycięcie fragmentu czasowego (od 10 s do 40 s) z próby nr 1:
    python -m ml.extract_sample --input EEG_preprocessed/1.mat --trial 1 --start 10 --end 40 --output data/sample_slice.mat

    # 4. Pocięcie długiego sygnału ciągłego na równe segmenty po 30 sekund:
    python -m ml.extract_sample --input duzy_sygnal.mat --segment-length 30 --output-dir data/segmenty/

    # 5. Lista dostępnych prób w pliku:
    python -m ml.extract_sample --input EEG_preprocessed/1.mat --list
"""

import argparse
from pathlib import Path

import scipy.io

from app.utils import (
    detect_format,
    list_trials,
    load_mat,
    normalize_signal_shape,
    slice_signal,
    split_mat_file,
)
from ml.config import SAMPLING_RATE


def main():
    parser = argparse.ArgumentParser(
        description="Narzędzie do wycinania, segmentacji i eksportowania próbek EEG do formatu .mat"
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Ścieżka do wejściowego pliku .mat")
    parser.add_argument("--trial", default=None,
                        help="Numer próby (np. 1) lub nazwa klucza (np. de_LDS_1)")
    parser.add_argument("--output", default=None, type=Path,
                        help="Ścieżka wyjściowa dla pojedynczego pliku .mat")
    parser.add_argument("--output-dir", default=None, type=Path,
                        help="Folder wyjściowy przy masowym podziale (--split-all lub --segment-length)")
    parser.add_argument("--split-all", action="store_true",
                        help="Automatycznie wyodrębnia wszystkie próby z pliku do osobnych plików .mat")
    parser.add_argument("--segment-length", type=float, default=None,
                        help="Długość segmentu w sekundach do pocięcia sygnału ciągłego")
    parser.add_argument("--start", type=float, default=None,
                        help="Początek wycinka czasowego w sekundach (dla sygnału)")
    parser.add_argument("--end", type=float, default=None,
                        help="Koniec wycinka czasowego w sekundach (dla sygnału)")
    parser.add_argument("--list", action="store_true",
                        help="Wypisuje dostępne próby w pliku i kończy działanie")

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Plik wejściowy nie istnieje: {args.input}")

    arrays = load_mat(args.input)
    trials = list_trials(arrays)

    if args.list:
        print(f"\nDostępne próby w pliku '{args.input}':")
        for label, (key, fmt) in trials.items():
            shape = arrays[key].shape
            print(f"  - {label:25s} | klucz: '{key}', format: {fmt}, kształt: {shape}")
        return

    # Tryb 1: Masowy podział na pojedyncze pliki
    if args.split_all or args.segment_length is not None:
        out_dir = args.output_dir or Path("data/extracted_samples") / args.input.stem
        print(f"Rozpoczynanie wyodrębniania do folderu: '{out_dir}'...")
        created = split_mat_file(args.input, out_dir, segment_length_sec=args.segment_length, fs=SAMPLING_RATE)
        print(f"Pomyślnie wyodrębniono {len(created)} plików prób do '{out_dir}'.")
        return

    # Tryb 2: Wyciągnięcie pojedynczej próby / fragmentu
    out_file = args.output or Path("data") / f"sample_{args.input.stem}.mat"

    if args.trial is None:
        first_label = list(trials.keys())[0]
        key, fmt = trials[first_label]
        print(f"Nie podano parametru --trial. Wybrano pierwszą dostępną próbę: '{key}'.")
    else:
        target = str(args.trial)
        key = None
        fmt = None
        for k in [target, f"de_LDS_{target}", f"de_{target}", f"psd_{target}"]:
            if k in arrays:
                key = k
                fmt = detect_format(arrays[k])
                break
        if key is None:
            raise KeyError(f"Nie znaleziono próby '{args.trial}' w pliku {args.input}. "
                           f"Użyj flagi --list, aby zobaczyć dostępne klucze.")

    data = arrays[key]
    fmt = detect_format(data)

    if fmt == "signal":
        data = normalize_signal_shape(data)
        if args.start is not None or args.end is not None:
            total_sec = data.shape[1] / SAMPLING_RATE
            start = 0.0 if args.start is None else args.start
            end = total_sec if args.end is None else args.end
            data = slice_signal(data, start, end, fs=SAMPLING_RATE)
            print(f"Wycięto fragment: [{start:.1f} s – {end:.1f} s], kształt: {data.shape}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(out_file), {key: data})
    print(f"Zapisano próbkę do: '{out_file}' (klucz: '{key}', kształt: {data.shape})")


if __name__ == "__main__":
    main()
