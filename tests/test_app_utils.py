"""Testy funkcji pomocniczych aplikacji: detekcja formatu, przycinanie sygnału, eksport i split."""

import numpy as np
import pytest
import scipy.io

from app.utils import (
    MatFileError,
    detect_format,
    export_to_mat_bytes,
    list_trials,
    load_mat,
    normalize_signal_shape,
    prepare_feature_vector,
    slice_signal,
    split_mat_file,
    validate_channel_names,
    validate_feature_values,
    validate_sampling_rate,
    validate_signal_properties,
)
from ml.config import SAMPLING_RATE, SEED_VII_CHANNELS


class TestDetectFormat:
    @staticmethod
    def signal(shape=(62, 12000)):
        return np.random.default_rng(42).normal(0, 15, size=shape)

    def test_cechy_3d(self):
        assert detect_format(np.zeros((15, 5, 62))) == "features"

    def test_cechy_2d(self):
        assert detect_format(np.zeros((5, 62))) == "features"

    def test_sygnal(self):
        assert detect_format(self.signal()) == "signal"

    def test_sygnal_transponowany(self):
        assert detect_format(self.signal().T) == "signal"

    def test_zly_format(self):
        with pytest.raises(MatFileError):
            detect_format(np.zeros((61, 12000)))
        with pytest.raises(MatFileError):
            detect_format(np.zeros((15, 4, 62)))
        with pytest.raises(MatFileError):
            detect_format(np.zeros(310))


class TestNormalizeSignalShape:
    def test_juz_poprawny(self):
        arr = np.zeros((62, 1000))
        assert normalize_signal_shape(arr).shape == (62, 1000)

    def test_transpozycja(self):
        arr = np.zeros((1000, 62))
        assert normalize_signal_shape(arr).shape == (62, 1000)


class TestSliceSignal:
    def test_poprawny_wycinek(self):
        # 10 sekund sygnału (2000 próbek)
        signal = np.arange(62 * 2000).reshape((62, 2000))
        sliced = slice_signal(signal, start_sec=2.0, end_sec=7.0, fs=SAMPLING_RATE)
        # 5 sekund = 1000 próbek
        assert sliced.shape == (62, 1000)
        assert sliced[0, 0] == signal[0, 400]

    def test_za_krotki_wycinek(self):
        signal = np.zeros((62, 2000))
        with pytest.raises(MatFileError):
            slice_signal(signal, start_sec=0.0, end_sec=2.0, fs=SAMPLING_RATE)  # 2s < 4s minimum

    def test_zly_zakres_czasu(self):
        signal = np.zeros((62, 2000))
        with pytest.raises(MatFileError):
            slice_signal(signal, start_sec=5.0, end_sec=4.0, fs=SAMPLING_RATE)
        with pytest.raises(MatFileError):
            slice_signal(signal, start_sec=-1.0, end_sec=5.0, fs=SAMPLING_RATE)
        with pytest.raises(MatFileError):
            slice_signal(signal, start_sec=0.0, end_sec=15.0, fs=SAMPLING_RATE)


class TestExportToMatBytes:
    def test_eksport_i_odczyt(self):
        import io
        arr = np.random.randn(62, 800)
        mat_bytes = export_to_mat_bytes(arr, key_name="test_key")
        assert isinstance(mat_bytes, bytes)
        assert len(mat_bytes) > 0

        # Weryfikacja załadowania z pamięci
        buf = io.BytesIO(mat_bytes)
        loaded = scipy.io.loadmat(buf)
        assert "test_key" in loaded
        np.testing.assert_array_almost_equal(loaded["test_key"], arr)


class TestSplitMatFile:
    def test_split_multi_trial(self, tmp_path):
        in_file = tmp_path / "multi_trial.mat"
        rng = np.random.default_rng(42)
        scipy.io.savemat(str(in_file), {
            "1": rng.normal(0, 15, size=(62, 1000)),
            "2": rng.normal(0, 15, size=(62, 1200)),
            "3": rng.normal(0, 15, size=(62, 1400)),
        })
        out_dir = tmp_path / "extracted"
        created = split_mat_file(in_file, out_dir)
        assert len(created) == 3
        assert (out_dir / "trial_1.mat").exists()
        assert (out_dir / "trial_2.mat").exists()
        assert (out_dir / "trial_3.mat").exists()

    def test_split_continuous_segments(self, tmp_path):
        in_file = tmp_path / "continuous.mat"
        # 60 sekund sygnału (12000 próbek @ 200 Hz)
        scipy.io.savemat(str(in_file), {
            "raw_signal": np.random.default_rng(42).normal(0, 15, size=(62, 12000))
        })
        out_dir = tmp_path / "segments"
        # Dzielenie na segmenty po 20 sekund -> 3 segmenty
        created = split_mat_file(in_file, out_dir, segment_length_sec=20.0, fs=SAMPLING_RATE)
        assert len(created) == 3
        assert (out_dir / "segment_1.mat").exists()
        assert (out_dir / "segment_2.mat").exists()
        assert (out_dir / "segment_3.mat").exists()


class TestListTrials:
    def test_preferuje_de_lds(self):
        arrays = {
            "de_1": np.zeros((15, 5, 62)),
            "de_LDS_1": np.zeros((15, 5, 62)),
            "psd_1": np.zeros((15, 5, 62)),
        }
        trials = list_trials(arrays)
        assert list(trials) == ["de_LDS_1 (features)"]

    def test_brak_poprawnych_tablic(self):
        with pytest.raises(MatFileError):
            list_trials({"x": np.zeros((3, 3))})

    def test_trials_values_format(self):
        arrays = {"de_LDS_1": np.zeros((15, 5, 62))}
        trials = list_trials(arrays)
        fmt = next(iter(trials.values()))[1]
        assert fmt == "features"

        arrays_sig = {"s01_0101": np.random.randn(62, 1000) * 15.0}
        trials_sig = list_trials(arrays_sig)
        fmt_sig = next(iter(trials_sig.values()))[1]
        assert fmt_sig == "signal"


class TestPrepareFeatureVector:
    def test_cechy(self):
        vec, feats = prepare_feature_vector(np.ones((15, 5, 62)), "features")
        assert vec.shape == (310,)
        assert feats.shape == (15, 5, 62)

    def test_cechy_jedno_okno_2d(self):
        vec, feats = prepare_feature_vector(np.ones((5, 62)), "features")
        assert vec.shape == (310,)
        assert feats.shape == (1, 5, 62)

    def test_sygnal(self):
        rng = np.random.default_rng(42)
        vec, feats = prepare_feature_vector(rng.normal(size=(62, 1600)), "signal")
        assert vec.shape == (310,)
        assert feats.shape == (2, 5, 62)

    def test_za_krotki_sygnal(self):
        with pytest.raises(MatFileError):
            prepare_feature_vector(np.zeros((62, 100)), "signal")

    def test_zla_liczba_kanalow(self):
        with pytest.raises(MatFileError):
            prepare_feature_vector(np.zeros((61, 1600)), "signal")


class TestValidationLayer:
    def test_odrzuca_sygnal_plaski(self):
        with pytest.raises(MatFileError, match="nienaturalnie niska"):
            validate_signal_properties(np.zeros((62, 1600)), key="flat_signal")

    def test_odrzuca_nan_i_inf(self):
        features = np.ones((10, 5, 62))
        features[0, 0, 0] = np.nan
        with pytest.raises(MatFileError, match="NaN"):
            validate_feature_values(features)

        signal = np.ones((62, 1600)) * 10
        signal[0, 0] = np.inf
        with pytest.raises(MatFileError, match="NaN"):
            validate_signal_properties(signal)

    def test_odrzuca_klucz_psd_w_detect_format(self):
        with pytest.raises(MatFileError, match="Wykryto cechy PSD"):
            detect_format(np.ones((10, 5, 62)), key="psd_1")

    def test_odrzuca_skale_psd_w_cechach(self):
        # Wartości typowe dla PSD (10^5) zamiast DE (~20)
        psd_array = np.full((10, 5, 62), 10000.0)
        with pytest.raises(MatFileError, match="gęstość widmową mocy"):
            detect_format(psd_array, key="unknown_features")

    def test_odrzuca_plik_z_samym_psd_w_list_trials(self):
        arrays = {
            "psd_1": np.full((10, 5, 62), 10000.0),
            "psd_2": np.full((10, 5, 62), 20000.0),
        }
        with pytest.raises(MatFileError, match="Wykryto wyłącznie tablice cech PSD"):
            list_trials(arrays)

    def test_walidacja_czestotliwosci_probkowania(self):
        assert validate_sampling_rate(200) == 200.0
        assert validate_sampling_rate(np.array([[200]])) == 200.0
        with pytest.raises(MatFileError, match="Niezgodna częstotliwość próbkowania"):
            validate_sampling_rate(250)
        with pytest.raises(MatFileError, match="Niezgodna częstotliwość próbkowania"):
            validate_sampling_rate(1000)

    def test_walidacja_kanalow_seed_vii(self):
        assert len(validate_channel_names(SEED_VII_CHANNELS)) == 62
        # Zamiana dwóch kanałów miejscami (np. Fp1 i Fpz)
        swapped = list(SEED_VII_CHANNELS)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        with pytest.raises(MatFileError, match="Niezgodność kolejności lub nazw"):
            validate_channel_names(swapped)
        # Zła liczba kanałów
        with pytest.raises(MatFileError, match="Nieprawidłowa liczba kanałów"):
            validate_channel_names(SEED_VII_CHANNELS[:60])

    def test_odrzuca_sygnal_w_woltach(self):
        # 0.00005 V (zamiast 50 uV)
        tiny_signal = np.full((62, 1600), 0.00005)
        with pytest.raises(MatFileError, match="woltach"):
            validate_signal_properties(tiny_signal, key="raw_volt_signal")

    def test_odrzuca_sygnal_z_dryftem_dc(self):
        # Średnia 500 uV (dryft stałoprądowy, brak filtra górnoprzepustowego)
        drift_signal = np.ones((62, 1600)) * 500.0 + np.random.randn(62, 1600) * 10.0
        with pytest.raises(MatFileError, match="dryft stałoprądowy"):
            validate_signal_properties(drift_signal, key="dc_drift_signal")

    def test_odrzuca_sygnal_z_nasyceniem(self):
        # std 3000 uV (potężne artefakty nasycenia przetwornika)
        saturated = np.random.randn(62, 1600) * 3000.0
        with pytest.raises(MatFileError, match="nasycenia"):
            validate_signal_properties(saturated, key="saturated_signal")

    def test_load_mat_weryfikuje_metadane(self, tmp_path):
        bad_file = tmp_path / "bad_fs.mat"
        scipy.io.savemat(str(bad_file), {
            "1": np.zeros((62, 1000)),
            "fs": 500,  # 500 Hz zamiast 200 Hz
        })
        with pytest.raises(MatFileError, match="Niezgodna częstotliwość próbkowania: 500"):
            load_mat(str(bad_file))

    def test_odrzuca_dane_okulograficzne_eye_features(self):
        # 33 cechy śledzenia wzroku
        eye_data = np.random.randn(20, 33)
        with pytest.raises(MatFileError, match="EYE_features"):
            detect_format(eye_data, key="1")

        with pytest.raises(MatFileError, match="EYE_features"):
            list_trials({"1": np.random.randn(20, 33), "2": np.random.randn(40, 33)})

    def test_akceptuje_transponowane_cechy_62_5(self):
        # Transponowany układ (okna, 62, 5) zamiast (okna, 5, 62)
        transposed = np.ones((10, 62, 5)) * 15.0
        assert detect_format(transposed, key="features_transposed") == "features"
        vec, feats = prepare_feature_vector(transposed, "features")
        assert vec.shape == (310,)
        assert feats.shape == (10, 5, 62)

        transposed_2d = np.ones((62, 5)) * 15.0
        assert detect_format(transposed_2d, key="de_LDS_1") == "features"
        vec_2d, feats_2d = prepare_feature_vector(transposed_2d, "features")
        assert vec_2d.shape == (310,)
        assert feats_2d.shape == (1, 5, 62)


class TestPdfReport:
    """Testy generowania raportu PDF."""

    def test_generuje_pdf_dla_sygnalu(self):
        from app.pdf_report import generate_pdf_report

        proba = np.array([0.98, 0.01, 0.005, 0.002, 0.001, 0.001, 0.001])
        classes = np.array([0, 1, 2, 3, 4, 5, 6])
        features = np.random.randn(10, 5, 62) * 5.0 + 15.0
        raw = np.random.randn(62, 2000) * 10.0

        pdf_bytes = generate_pdf_report("sample_signal.mat", "signal", raw, features, proba, classes)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(pdf_bytes) > 20000

    def test_generuje_pdf_dla_cech(self):
        from app.pdf_report import generate_pdf_report

        proba = np.array([0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05])
        classes = np.array([0, 1, 2, 3, 4, 5, 6])
        features = np.random.randn(15, 5, 62) * 5.0 + 15.0

        pdf_bytes = generate_pdf_report("sample_de_features.mat", "features", None, features, proba, classes)
        assert isinstance(pdf_bytes, bytes)
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(pdf_bytes) > 20000
