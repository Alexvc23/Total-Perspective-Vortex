import mne
from mne.datasets import eegbci, fetch_fsaverage
from mne.io import read_raw_edf

# 1. Force the engine to use a native macOS window
mne.viz.set_3d_backend("pyvistaqt")

print("1. Loading EEG Data...")
# Load Subject 1, Run 4 to get the channel information
raw_fnames = eegbci.load_data(
    1, [4], path="./doc/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0", verbose=False
)
raw = read_raw_edf(raw_fnames[0], preload=False, verbose=False)

# Apply the standard 10-20 montage
eegbci.standardize(raw)
raw.set_montage("standard_1005", match_case=False)

print("2. Fetching Anatomical Template...")
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = fs_dir.parent

print("3. Rendering 3D Window (Look for a new icon in your Mac Dock!)...")
# Render the 3D alignment
fig = mne.viz.plot_alignment(
    raw.info,
    subject="fsaverage",
    subjects_dir=subjects_dir,
    trans="fsaverage",
    surfaces="head",
    show_axes=False,
    eeg="projected",
    coord_frame="mri",
)

# Set a nice default camera angle
mne.viz.set_3d_view(figure=fig, azimuth=135, elevation=80)

# 4. CRITICAL FOR SCRIPTS: Pause execution
# If we don't add this, Python will finish reading the file and instantly close the window!
input(
    "\n✅ Success! The 3D window is now open. Press [ENTER] here in the terminal to close the window and exit..."
)
