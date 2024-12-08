import re
import numpy as np
import librosa
from mutagen.easyid3 import EasyID3

SAMPLE_RATE = 44100
HOP_LENGTH = 512

def clean_string(value):
    value = value.lower() 
    value = re.sub(r'[^a-z0-9]', '_', value) 
    value = re.sub(r'_+', '_', value) 
    value = value.strip('_')
    return value

class TrackAnalyzer:
    def __init__(self, filename):
        self.filename = filename
        self.artist = None
        self.title = None
        self.audio = None
        self.sample_rate = None
        self.tempo = None
        self.beat_frames = None
        self.rms_mean = None
        self.rms_std = None
        self.cent_mean = None
        self.cent_std = None
        self.chroma_stats = None
        self.mfcc_stats = None
        self.bandwidth_mean = None
        self.bandwidth_std = None
        self.contrast_stats = None
        self.flatness_mean = None
        self.flatness_std = None
        self.rolloff_mean = None
        self.rolloff_std = None
        self.tonnetz_stats = None
        self.zcr_mean = None
        self.zcr_std = None
        self.tempogram_ratio_stats = None
        self.onset_strength_mean = None
        self.onset_strength_std = None
        self.dynamic_range = None

        self._load_metadata()
        self._load_audio_data()

    def extract_features(self):
        """Extract all requested audio features."""
        self._calculate_tempo_and_beats()
        self._calculate_rms()
        self._calculate_spectral_centroid()
        self._calculate_chroma_features()
        self._calculate_mfcc_features()
        self._calculate_spectral_bandwidth()
        self._calculate_spectral_contrast()
        self._calculate_spectral_flatness()
        self._calculate_spectral_rolloff()
        self._calculate_tonnetz()
        self._calculate_zcr()
        self._calculate_tempogram_ratio()
        self._calculate_onset_strength()
        self._calculate_dynamic_range()

    def print_info(self):
        """Print all extracted metadata and audio features."""
        print(f"Artist: {self.artist}, Title: {self.title}, Tempo: {self.tempo}")
        print(f"RMS: {self.rms_mean:.4f} (mean), {self.rms_std:.4f} (std)")
        print(f"Spectral centroid: {self.cent_mean:.4f} (mean), {self.cent_std:.4f} (std)")

        if self.chroma_stats:
            for note, stats in self.chroma_stats.items():
                print(f"{note} (mean): {stats['mean']:.4f}, {note} (std): {stats['std']:.4f}")
        
        if self.mfcc_stats:
            for idx, stats in self.mfcc_stats.items():
                print(f"MFCC-{idx + 1} (mean): {stats['mean']:.4f}, MFCC-{idx + 1} (std): {stats['std']:.4f}")

        if self.bandwidth_mean is not None:
            print(f"Spectral Bandwidth (mean): {self.bandwidth_mean:.4f}, (std): {self.bandwidth_std:.4f}")

        if self.contrast_stats is not None:
            for i, stats in self.contrast_stats.items():
                print(f"Spectral Contrast Band {i+1} (mean): {stats['mean']:.4f}, (std): {stats['std']:.4f}")

        if self.flatness_mean is not None:
            print(f"Spectral Flatness (mean): {self.flatness_mean:.4f}, (std): {self.flatness_std:.4f}")

        if self.rolloff_mean is not None:
            print(f"Spectral Rolloff (mean): {self.rolloff_mean:.4f}, (std): {self.rolloff_std:.4f}")

        if self.tonnetz_stats is not None:
            for i, stats in self.tonnetz_stats.items():
                print(f"Tonnetz Dimension {i+1} (mean): {stats['mean']:.4f}, (std): {stats['std']:.4f}")

        if self.zcr_mean is not None:
            print(f"ZCR (mean): {self.zcr_mean:.4f}, (std): {self.zcr_std:.4f}")

        if self.tempogram_ratio_stats is not None:
            for i, stats in self.tempogram_ratio_stats.items():
                print(f"Tempogram Ratio Factor {i+1} (mean): {stats['mean']:.4f}, (std): {stats['std']:.4f}")

        if self.onset_strength_mean is not None:
            print(f"Onset Strength (mean): {self.onset_strength_mean:.4f}, (std): {self.onset_strength_std:.4f}")

        if self.dynamic_range is not None:
            print(f"Dynamic Range (dB): {self.dynamic_range:.4f}")

    def _load_metadata(self):
        """Load artist, title and genre metadata from the file."""
        try:
            audio_tags = EasyID3(self.filename)
            self.artist = clean_string(audio_tags.get("artist")[0])
            self.title = clean_string(audio_tags.get("title")[0])
            self.genre = clean_string(audio_tags.get("genre")[0])
        except Exception as e:
            print(f"Error reading metadata: {e}")

    def _load_audio_data(self):
        """Load audio data from the file."""
        try:
            self.audio, self.sample_rate = librosa.load(self.filename, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading audio: {e}")

    def _calculate_tempo_and_beats(self):
        """Calculate tempo and beat frames."""
        tempo, beat_frames = librosa.beat.beat_track(
            y=self.audio, sr=self.sample_rate, hop_length=HOP_LENGTH
        )
        if isinstance(tempo, np.ndarray) and tempo.size == 1:
            tempo = tempo.item() 
        self.tempo = float(tempo)  # Ensure it's a scalar float, was getting deprecated warning
        self.beat_frames = beat_frames

    def _calculate_rms(self):
        """Calculate beat-synchronized RMS."""
        rms = librosa.feature.rms(y=self.audio)
        rms_sync = librosa.util.sync(rms, self.beat_frames, aggregate=np.mean)
        self.rms_mean = np.mean(rms_sync)
        self.rms_std = np.std(rms_sync)

    def _calculate_spectral_centroid(self):
        """Calculate beat-synchronized spectral centroid."""
        cent = librosa.feature.spectral_centroid(y=self.audio, sr=self.sample_rate)
        cent_sync = librosa.util.sync(cent, self.beat_frames, aggregate=np.mean)
        self.cent_mean = np.mean(cent_sync)
        self.cent_std = np.std(cent_sync)

    def _calculate_chroma_features(self):
        """Calculate beat-synchronized chroma features and store their stats."""
        chroma = librosa.feature.chroma_cqt(y=self.audio, sr=self.sample_rate, hop_length=HOP_LENGTH)
        chroma_sync = librosa.util.sync(chroma, self.beat_frames, aggregate=np.mean)

        means = np.mean(chroma_sync, axis=1)
        stds = np.std(chroma_sync, axis=1)

        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        self.chroma_stats = {
            note: {"mean": m, "std": s}
            for note, m, s in zip(note_names, means, stds)
        }

    def _calculate_mfcc_features(self):
        """Calculate beat-synchronized MFCC features and store their stats."""
        mfcc = librosa.feature.mfcc(y=self.audio, sr=self.sample_rate, n_mfcc=20, hop_length=HOP_LENGTH)
        mfcc_sync = librosa.util.sync(mfcc, self.beat_frames, aggregate=np.mean)

        means = np.mean(mfcc_sync, axis=1)
        stds = np.std(mfcc_sync, axis=1)

        self.mfcc_stats = {
            idx: {"mean": m, "std": s}
            for idx, (m, s) in enumerate(zip(means, stds))
        }

    def _calculate_spectral_bandwidth(self):
        sb = librosa.feature.spectral_bandwidth(y=self.audio, sr=self.sample_rate, hop_length=HOP_LENGTH)
        sb_sync = librosa.util.sync(sb, self.beat_frames, aggregate=np.mean)
        self.bandwidth_mean = np.mean(sb_sync)
        self.bandwidth_std = np.std(sb_sync)

    def _calculate_spectral_contrast(self):
        sc = librosa.feature.spectral_contrast(y=self.audio, sr=self.sample_rate, hop_length=HOP_LENGTH)
        sc_sync = librosa.util.sync(sc, self.beat_frames, aggregate=np.mean)
        means = np.mean(sc_sync, axis=1)
        stds = np.std(sc_sync, axis=1)

        self.contrast_stats = {
            i: {"mean": m, "std": s}
            for i, (m, s) in enumerate(zip(means, stds))
        }

    def _calculate_spectral_flatness(self):
        sf = librosa.feature.spectral_flatness(y=self.audio, hop_length=HOP_LENGTH)
        sf_sync = librosa.util.sync(sf, self.beat_frames, aggregate=np.mean)
        self.flatness_mean = np.mean(sf_sync)
        self.flatness_std = np.std(sf_sync)

    def _calculate_spectral_rolloff(self):
        sro = librosa.feature.spectral_rolloff(y=self.audio, sr=self.sample_rate, hop_length=HOP_LENGTH)
        sro_sync = librosa.util.sync(sro, self.beat_frames, aggregate=np.mean)
        self.rolloff_mean = np.mean(sro_sync)
        self.rolloff_std = np.std(sro_sync)

    def _calculate_tonnetz(self):
        # Tonnetz requires chroma. If not pre-computed, librosa will compute internally.
        # We already have chroma extracted, but we can pass our audio and sr directly:
        tn = librosa.feature.tonnetz(y=self.audio, sr=self.sample_rate)
        tn_sync = librosa.util.sync(tn, self.beat_frames, aggregate=np.mean)
        means = np.mean(tn_sync, axis=1)
        stds = np.std(tn_sync, axis=1)
        self.tonnetz_stats = {
            i: {"mean": m, "std": s}
            for i, (m, s) in enumerate(zip(means, stds))
        }

    def _calculate_zcr(self):
        zcr = librosa.feature.zero_crossing_rate(y=self.audio, hop_length=HOP_LENGTH)
        zcr_sync = librosa.util.sync(zcr, self.beat_frames, aggregate=np.mean)
        self.zcr_mean = np.mean(zcr_sync)
        self.zcr_std = np.std(zcr_sync)

    def _calculate_tempogram_ratio(self):
        # Compute tempogram first
        tg = librosa.feature.tempogram(y=self.audio, sr=self.sample_rate, hop_length=HOP_LENGTH)

        # Use pre-computed tempo as bpm
        tgr = librosa.feature.tempogram_ratio(tg=tg, sr=self.sample_rate, bpm=self.tempo, hop_length=HOP_LENGTH)

        # tgr shape: (factors, frames)
        tgr_sync = librosa.util.sync(tgr, self.beat_frames, aggregate=np.mean)
        means = np.mean(tgr_sync, axis=1)
        stds = np.std(tgr_sync, axis=1)

        self.tempogram_ratio_stats = {
            i: {"mean": m, "std": s}
            for i, (m, s) in enumerate(zip(means, stds))
        }

    def _calculate_onset_strength(self):
        onset_env = librosa.onset.onset_strength(y=self.audio, sr=self.sample_rate, hop_length=HOP_LENGTH)
        # onset_env is 1D: shape=(frames,)
        # Make it 2D for sync: shape=(1, frames)
        onset_env = onset_env[np.newaxis, :]
        onset_sync = librosa.util.sync(onset_env, self.beat_frames, aggregate=np.mean)
        self.onset_strength_mean = np.mean(onset_sync)
        self.onset_strength_std = np.std(onset_sync)

    def _calculate_dynamic_range(self):
        peak_amplitude = np.max(np.abs(self.audio))
        # Use RMS mean from previously calculated RMS
        # Dynamic range in dB: 20 * log10(peak / rms_mean)
        # Assuming rms_mean > 0 (we assume data in good shape)
        if self.rms_mean is not None and self.rms_mean > 0:
            self.dynamic_range = 20 * np.log10(peak_amplitude / self.rms_mean)
        else:
            self.dynamic_range = None


# Example:
# analyzer = TrackAnalyzer("Cloonee - What Ya Want (Extended).mp3")
# analyzer.extract_features()
# analyzer.print_info()

def write_features_to_csv(analyzer, filename="track_features.csv"):
    columns = []
    values = []

    # Add metadata
    columns.append("artist")
    values.append(analyzer.artist if analyzer.artist else "")
    columns.append("title")
    values.append(analyzer.title if analyzer.title else "")
    columns.append("genre")
    values.append(analyzer.genre if analyzer.artist else "")
    columns.append("tempo")
    values.append(analyzer.tempo if analyzer.tempo is not None else "")

    # Add scalar features
    scalar_features = {
        "rms_mean": analyzer.rms_mean,
        "rms_std": analyzer.rms_std,
        "spectral_centroid_mean": analyzer.cent_mean,
        "spectral_centroid_std": analyzer.cent_std,
        "spectral_bandwidth_mean": analyzer.bandwidth_mean,
        "spectral_bandwidth_std": analyzer.bandwidth_std,
        "spectral_flatness_mean": analyzer.flatness_mean,
        "spectral_flatness_std": analyzer.flatness_std,
        "spectral_rolloff_mean": analyzer.rolloff_mean,
        "spectral_rolloff_std": analyzer.rolloff_std,
        "zcr_mean": analyzer.zcr_mean,
        "zcr_std": analyzer.zcr_std,
        "onset_strength_mean": analyzer.onset_strength_mean,
        "onset_strength_std": analyzer.onset_strength_std,
        "dynamic_range_db": analyzer.dynamic_range
    }

    for name, val in scalar_features.items():
        columns.append(name)
        values.append(val if val is not None else "")

    # Add chroma features (12 notes)
    if analyzer.chroma_stats is not None:
        for note, stats in analyzer.chroma_stats.items():
            columns.append(f"chroma_{note}_mean")
            values.append(stats["mean"])
            columns.append(f"chroma_{note}_std")
            values.append(stats["std"])

    # Add MFCC features (20 coefficients)
    if analyzer.mfcc_stats is not None:
        for i, stats in analyzer.mfcc_stats.items():
            idx = i + 1
            columns.append(f"mfcc_{idx}_mean")
            values.append(stats["mean"])
            columns.append(f"mfcc_{idx}_std")
            values.append(stats["std"])

    # Add spectral contrast features (7 bands)
    if analyzer.contrast_stats is not None:
        for i, stats in analyzer.contrast_stats.items():
            band = i + 1
            columns.append(f"spectral_contrast_band{band}_mean")
            values.append(stats["mean"])
            columns.append(f"spectral_contrast_band{band}_std")
            values.append(stats["std"])

    # Add tonnetz features (6 dimensions)
    if analyzer.tonnetz_stats is not None:
        for i, stats in analyzer.tonnetz_stats.items():
            dim = i + 1
            columns.append(f"tonnetz_{dim}_mean")
            values.append(stats["mean"])
            columns.append(f"tonnetz_{dim}_std")
            values.append(stats["std"])

    # Add tempogram ratio features
    if analyzer.tempogram_ratio_stats is not None:
        for i, stats in analyzer.tempogram_ratio_stats.items():
            factor = i + 1
            columns.append(f"tempogram_ratio_factor{factor}_mean")
            values.append(stats["mean"])
            columns.append(f"tempogram_ratio_factor{factor}_std")
            values.append(stats["std"])

    # Write to CSV file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(",".join(str(col) for col in columns) + "\n") # TODO - will need to write header column seperatly
        f.write(",".join(str(val) for val in values) + "\n")


# Example usage:
analyzer = TrackAnalyzer("Cloonee - What Ya Want (Extended).mp3")
analyzer.extract_features()
write_features_to_csv(analyzer, "track_features.csv")
