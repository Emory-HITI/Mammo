# Mammo
Various code for mammography data (including mammography image type derivation and ROI extraction)

MammoROI

Description:
The MammoROI code is used to identify image type, special views, and extract ROI coordinates of images from cohorts

Libraries:
dataframe_processing - functions for dataframe manipulations
roi_extractions - functions for roi extraction

dataframe_processing Functions:

load_simple_df(): loads metadata csvs as pandas dataframes with only necessary columns

Input(s): df_path, df_type=None
df_path: path to cohort csv  e.g.: ‘/mnt/PACSNAS/…/...cohort_X.csv’ 
metadata_anon_cohort_X.csv
All anonymized data
metadata_orig_and_anon_cohort_X.csv
Same as metadata_anon_cohort_X.csv but contains original PHI information
df_type: default None, optional ‘ROI_merge’, ‘ROI_extract’
None: reads just the columns necessary for image type and special view derivations
‘ROI_extract’: reads just the columns necessary for ROI extraction
‘ROI_merge’: reads just the columns necessary for ROI dataframe merging
Output(s): 
loaded pandas dataframe on a specified variable
Sample Usage:
cohort_1_df = load_simple_df(‘/path/.../metadata_anon_cohort_1 .csv’)

correct_root_paths(): replaces png_path roots to correct ones (if needed)

Input(s): dataframe, root_a, root_b
dataframe: loaded pandas dataframe with file paths
column ‘png_path’
root_a: root path to be replacee
e.g. ‘/opt/ssd-data/’
root_b: root path to replace with
e.g. ‘/mnt/PACS_NAS1/’
Output(s): Pandas dataframe with new columns [‘corrected_png_path’, ‘folder_path’, ‘filename’]
‘corrected_png_path’: new file paths with root_a changed to root_b
‘folder_path’: folder path to the accession/instance
‘filename’: png filename
Sample Usage:
cohort_1_df = correct_paths_PACS(cohort_1_df, ‘/opt/ssd-data/’, ‘/mnt/PACS_NAS1/’)

derive_imgType(): derives/identifies image types and lateralities 
Input(s): df_in
df_in: dataframe with columns [‘SeriesDescription’ , ‘ImageLaterality’]
Loaded variable with load_simple_df(‘/path/.../metadata_anon_cohort_1 .csv’)
Output(s): df_out 
df_out: dataframe with new columns added [‘LateralityDeriveFlag’, ‘ImageLateralityFinal’, ‘FinalImageType’]
‘LateralityDeriveFlag’: was the laterality derived from the series description with the code or was laterality already available 
0 or 1
‘ImageLateralityFinal’: the final laterality
L or R
No bilaterals (‘B’)
‘FinalImageType’: the final image type 
cview, 2D, 3D, ROI_SSC, ROI_SS, other 
Other includes bilateral types
Sample Usage:
cohort_1_df = derive_imgType(cohort_1_df)

get_spotmag_flags(): derives/identifies special images
Input(s): df_in
df_in: dataframe with columns [‘0_ViewCodeSequence_0_ViewModifierCodeSequence_CodeMeaning’]
Loaded variable with load_simple_df(‘/path/.../metadata_anon_cohort_1 .csv’)
Output(s): df_out 
df_out: dataframe with new column added [‘spot_mag’]
‘spot_mag’: is the image a special view or not
1 or NaN
Sample Usage:
cohort_1_df = get_spotmag_flags(cohort_1_df)

match_roi(): matches the correct ROIs to the actual mammogram
Input(s): main_df, roi_df
main_df: dataframe with metadata columns and [‘png_path’]
roi_df: output dataframe from ROI extraction with column [‘Matching_Mammo’, ‘ROI_coord’]
Output(s): df_out 
df_out: dataframe with new column added [‘ROI_coord’]
‘ROI_coord’: is the ROI coordinates of the corresponding mammogram or screen save in their size
[1251, 1536, 1354, 1856] (one list) for one ROI OR
[[154, 1567, 264, 1897], [1251, 1536, 1354, 1856]] list of lists for multiple.
Sample Usage:
main_df= match_roi(main_df, roi_df)

replace_old_png_path(): removes and replaces the ‘png_path’ with the correct path
Input(s): df_in
df_in: dataframe with columns [‘png_path’] and [‘corrected_png_path’]
Output(s): df_out 
df_out: dataframe with [‘png_path’] replaced with [‘corrected_png_path’]’s values
Sample Usage:
cohort_1_df = replace_old_png_path(cohort_1_df)

make_screensave_dict(): makes a dictionary of screensave filenames and paths for ROI extraction
Input(s): df_in
df_in: dataframe with columns [‘ROI_SSC’], [‘folder_path’],  and [‘filename’]
Output(s):out_dict
out_dict: dictionary with key value pairs of {filename: folder_path}
Sample Usage:
out_dict = make_screensave_dict(ROI_SSC_df)

read_df(): reads the ‘final’ metadata csv and loads the ROI_coord column as a list not a string
Input(s): path_to_csv
path_to_csv: path to the final metadata csv with all metadata with the lateralities, spotmags, and merged ROI coordinates
Output(s): df_out 
df_out: with final metadata
Sample Usage:
cohort_1_df = read_df(‘./.../.../metadata_cohort_1_ROI.csv’)

Tensorflow Object Detection Install: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

Installing Object Detection API:
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#downloading-the-tensorflow-model-garden

Install protobuf:
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#protobuf-installation-compilation

In the end, a root folder should have:
‘models’  folder: created from the object detection api install
‘OD_Files’ folder: a provided folder with parameters for object detection model
‘training’ folder: a provided folder with the trained model and checkpoints


roi_extractions Functions:
First need to start a new instance of the class:
ROI = ROI_extraction(‘.../.../path to root directory with tensorflow object detections)
e.g.: ROI = ROI_extraction('/home/jupyter-jjjeon3')

run_extractions(): loads metadata csvs as pandas dataframes with only necessary columns

Input(s): ss_dict
ss_dict: dictionary output from make_screensave_dict() from dataframe_processing 
Output(s): 
ROI_coords_ssc, ROI_coords_mammo, ROI_matching_ssc, ROI_matching_mammo
ROI_coords_ssc: list of screen save coordinates (in screen save size)
ROI_coords_mammo: list of mammography coordinates (in mammography size and flipped accordingly)
ROI_matching_ssc: the path to the screen save image for the same index of the ROI_coords_ssc
ROI_matching_mammo: the path to the mammogram image for the same index of the ROI_coords_ssc
Sample Usage:
cohort_1_df = load_simple_df(‘/path/.../metadata_anon_cohort_1 .csv’)

