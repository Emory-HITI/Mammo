import numpy as np
import pandas as pd

def derive_imgType(dataframe):
    """
    Extracts final image types and lateralities from the defined lists. Adjusts for the different naming convention of GE scanners.
    (updated: 7/30/2021)
    
    Sample Usage:
    df_new = derive_imgType(df)
    """
    # initializing empty lists:
    DeriveFlag = []  # ImageLaterality: 0 (not derived), 1 (derived), 2 (need to derive)
    ImageLateralityFinal = []  # the image laterality either copied or taken from series description
    FinalImageType = []  # final image type: 2D, 3D, cview, ROI_SSC, ROI_SS, Other (not any of the other ones)

    for index, row in dataframe.iterrows():  # iterating over all rows
        try:
            if type(row['SeriesDescription']) == str:
                if ' ' in row['SeriesDescription']:
                    desc = row['SeriesDescription'].split(' ')  # splitting up the series description for checking
                else:
                    desc = row['SeriesDescription'].split('_')  # GE tags
            else:
                desc = str(row['SeriesDescription'])  # filter for nan
        except:
            DeriveFlag.append(2)  # update derive flag = 0 
            ImageLateralityFinal.append(np.nan)  # update final ImageLaterality
            FinalImageType.append('other')  # update imagetype
            continue
        if row['ImageLaterality'] in ['L', 'R']:  # if lat exists, it's either 2D or cview or ROI
            if (('C-View' in desc or 'CV' in desc)):  # if it has C-View or CV it's cview
                DeriveFlag.append(0)  # update derive flag = 0 
                ImageLateralityFinal.append(row['ImageLaterality'])  # update final ImageLaterality
                FinalImageType.append('cview')  # update imagetype
            elif (('Tomosynthesis' in desc) or ('w/Tomosynthesis' in desc)):  # make sure all tomo is accounted for
                DeriveFlag.append(1)  # update derive flag = 1
                ImageLateralityFinal.append(row['ImageLaterality'])  # update final ImageLaterality
                FinalImageType.append('3D')  # update imagetype
            else:  # if it has lat and is not cview, it's 2d
                DeriveFlag.append(0)  # update derive flag = 0 
                ImageLateralityFinal.append(row['ImageLaterality'])  # update final ImageLaterality
                FinalImageType.append('2D')  # update imagetype
        else:  # if lat doesn't exist, it's either 3D, ROI
            if (('Tomosynthesis' in desc) or ('w/Tomosynthesis' in desc)):  # if it has Tomosynthesis Reconstruction is 3D
                DeriveFlag.append(1)  # update derive flag = 1
                ImageLateralityFinal.append(desc[0])  # update final ImageLaterality
                FinalImageType.append('3D')  # update imagetype
            elif 'ROUTINE3D' in desc:  # adding more flags for GE
                DeriveFlag.append(1)  # update derive flag = 1
                ImageLateralityFinal.append(desc[2][0])  # update final ImageLaterality
                FinalImageType.append('3D')  # update imagetype
            elif 'C-View' in desc:  # adding more flags for GE
                DeriveFlag.append(1)  # update derive flag = 1
                ImageLateralityFinal.append(desc[0])  # update final ImageLaterality
                FinalImageType.append('cview')  # update imagetype
            elif 'Capture' in desc:  # if it has Secondary Capture, it's SecurView Secondary Capture ROI
                DeriveFlag.append(2)  # update derive flag = 1
                ImageLateralityFinal.append(np.nan)  # update final ImageLaterality
                FinalImageType.append('ROI_SSC')  # update imagetype
            elif (('Screen' in desc) or ('SC' in desc)):
                DeriveFlag.append(2)  # update derive flag = 1
                ImageLateralityFinal.append(np.nan)  # update final ImageLaterality
                FinalImageType.append('ROI_SS')  # update imagetype
            else:
                DeriveFlag.append(2)  # update derive flag = 1
                ImageLateralityFinal.append(np.nan)  # update final ImageLaterality
                FinalImageType.append('other')  # update imagetype
            
    dataframe['LateralityDeriveFlag'] = DeriveFlag
    dataframe['ImageLateralityFinal'] = ImageLateralityFinal
    dataframe['FinalImageType'] = FinalImageType
    
    return dataframe

def get_spotmag_flags(df):
    """
    Extracts spotmag/special views from the specific tag. (weirdly enough, GE maintains the metadata for spotmag the same as others)
    
    Sample Usage:
    df_new = get_spotmag_flags(df)
    """
    spot_mags = []
    for i in range(len(df)):
        if df['0_ViewCodeSequence_0_ViewModifierCodeSequence_CodeMeaning'][i] in ['Spot Compression', 'Magnification']:
            spot_mags.append(1)
        else:
            spot_mags.append(np.nan)
    df['spot_mag'] = spot_mags
    return df

def correct_paths_PACS_hardcode(dataframe):
    """
    Hardcoded path correction (/opt/ssd-data/ to /mnt/PACS_NAS1/)and path splitting (png_path to path and filename)
    
    Sample Usage:
    df_new = correct_paths_PACS_hardcode(df)
    """
    correct_png_path = []
    for i in dataframe.png_path.to_list():
        path = i.strip() # remove any whitespace
        path = path.replace('//', '/')  #remove any double //
        path = path.replace('/home', '') # remove /home
        path = path.strip() # remove any whitespace
        path = path.replace('/opt/ssd-data/', '/mnt/PACS_NAS1/') # remove the opt/ssd-data
        #path = path.replace('png_kheiron/cohort_1', 'png_kheiron/kheiron_code/cohort_1')
        correct_png_path.append(path)
    dataframe['png_path'] = correct_png_path

    path = []
    filename = []
    for i in dataframe.png_path.to_list():
        # kheiron cohort csv has weird paths, need to correct a bit to match the hiti server
        path.append('/'.join(i.split('/')[:-1]))
        filename.append(i.split('/')[-1])
    dataframe['path'] = path
    dataframe['filename'] = filename
    
    return dataframe

def correct_paths_PACS(dataframe, root_a, root_b):
    """
    Futureproofing with input path correction code. Input root and output roots are given and paths corrected. 
    Path splitting (png_path to path and filename) still the same implementation.
    
    Sample Usage:
    df_new = correct_paths_PACS_hardcode(df, '/opt/ssd-data/', '/mnt/PACS_NAS1/')
    """
    correct_png_path = []
    for i in dataframe.png_path.to_list():
        path = i.strip() # remove any whitespace
        path = path.replace('//', '/')  #remove any double //
        path = path.replace('/home', '') # remove /home
        path = path.strip() # remove any whitespace
        path = path.replace(root_a, root_b) # remove the opt/ssd-data
        correct_png_path.append(path)
    dataframe['corrected_png_path'] = correct_png_path

    path = []
    filename = []
    for i in dataframe.corrected_png_path.to_list():
        # kheiron cohort csv has weird paths, need to correct a bit to match the hiti server
        path.append('/'.join(i.split('/')[:-1]))
        filename.append(i.split('/')[-1])
    dataframe['folder_path'] = path
    dataframe['filename'] = filename
    
    return dataframe

def match_roi(main_df, roi_df):
    """
    gets the main dataframe and ROI dataframe and merges the correct ROIs to the right column on the main dataframe
    """
    # gets the string path of the matching mammo
    roi_df['Matching_Mammo'] = roi_df['Matching_Mammo'].apply(lambda x: eval(x)[0] if eval(x) else  "" )
    mammo_roi = []
    # going through all the main df and checking for ROI df if ROI exists
    for i in range(len(main_df)):
        roi = roi_df['ROI_coord'][roi_df['Matching_Mammo'] == main_df['corrected_png_path'].iloc[i]]
        if roi.empty:
            mammo_roi.append([])
        else:
            mammo_roi.append([eval(t)[0] for t in roi ])  # converts ROI string into ROI coordinates
    main_df['ROI_coord'] = mammo_roi
    
    return main_df

def load_simple_df(df_path, df_type=None):
    if df_type == 'ROI':
        df = pd.read_csv(df_path, dtype='str')
        df = df[['png_path', 'Matching_Mammo', 'ROI_coord']]
    else:
        df = pd.read_csv(df_path, low_memory=False)
        df = df[['SeriesDescription', 'ImageLaterality', '0_ViewCodeSequence_0_ViewModifierCodeSequence_CodeMeaning', 'png_path']]
    return df
