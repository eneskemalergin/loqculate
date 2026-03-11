from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from lmfit import Minimizer, Parameters
from lmfit.models import LinearModel

DEFAULT_MIN_LINEAR_POINTS = 1
DEFAULT_MIN_NOISE_POINTS = 2

plt.style.use('seaborn-v0_8-whitegrid')

# Note: global np.random.seed removed; each bootstrap replicate uses an
# independent SeedSequence for thread-safe, reproducible resampling.
# Note: std_noise in calculate_lod uses ddof=1 (sample std, Bessel-corrected).
# np.std() defaults to ddof=0; ddof=1 is analytically correct for estimating
# a population noise parameter from a small sample (consistent with loq_by_cv.py).

# Force warnings (other than FutureWarning) to kill the script; this allows debugging numpy warnings.
#import warnings
#warnings.simplefilter("error")
#warnings.simplefilter("ignore", FutureWarning)
#warnings.filterwarnings("ignore")

# Assume input file source, read in the data, and return a standard-format melted dataframe
# output of this function is a melted dataframe with columns: peptide, curvepoint, area
def _validate_conc_map(col_conc_map_file):
    """Warn if the concentration map is empty or contains unannotated rows."""
    try:
        _map = pd.read_csv(col_conc_map_file)
    except Exception:
        return  # let the main read raise its own error
    if _map.empty or 'concentration' not in _map.columns:
        sys.stderr.write(
            'WARNING: the concentration map appears to be blank or is missing '
            'a "concentration" column. No curve points will be mapped.\n'
        )
        return
    blank = _map['concentration'].isna() | (_map['concentration'].astype(str).str.strip() == '')
    n_blank = int(blank.sum())
    if n_blank:
        sys.stderr.write(
            f'WARNING: {n_blank} row(s) in the concentration map have a blank/unannotated '
            f'concentration value and will be skipped:\n'
        )
        fname_col = 'filename' if 'filename' in _map.columns else _map.columns[0]
        for fname in _map.loc[blank, fname_col].tolist():
            sys.stderr.write(f'  {fname}\n')


def read_input(filename, col_conc_map_file):
    _validate_conc_map(col_conc_map_file)

    with open(filename, 'r') as f:
        header_line = f.readline()

    # if numFragments is a column, it's an Encyclopedia file
    if 'numFragments' in header_line:
        sys.stdout.write('Input assumed to be EncyclopeDIA *.elib.peptides.txt filetype.\n')

        df = pd.read_csv(filename, sep=None, engine='python')
        df.drop(['numFragments', 'Protein'], axis="columns", inplace=True)  # make a quantitative df with just curve points and peptides

        col_conc_map = pd.read_csv(col_conc_map_file, index_col="filename")

        # map filenames to concentrations
        df.rename(columns=dict(
            col_conc_map['concentration'],
            **{'Peptide': 'peptide'}
        ), inplace=True)

        df_melted = pd.melt(df, id_vars=['peptide'])
        df_melted.columns = ['peptide', 'curvepoint', 'area']
        df_melted = df_melted[df_melted['curvepoint'].isin(col_conc_map['concentration'])]

    # if Skyline file, require columns for File Name, Total Area Fragment, Peptide Sequence
    # TODO: option for Total Area Ratio?
    elif all(col in header_line for col in ['Total Area Fragment', 'Peptide Sequence', 'File Name']):
        sys.stdout.write('Input assumed to be Skyline export filetype. \n')

        df_melted = pd.read_csv(filename)
        df_melted.rename(columns={'File Name': 'filename'}, inplace=True)
        col_conc_map = pd.read_csv(col_conc_map_file)

        # remove any data for which there isn't a map key
        df_melted = df_melted[df_melted['filename'].isin(col_conc_map['filename'])]

        # map filenames to concentrations
        df_melted = pd.merge(df_melted, col_conc_map, on='filename', how='outer')

        # clean up column names to match downstream convention
        df_melted.rename(columns={'Total Area Fragment': 'area',
                                  'Peptide Sequence': 'peptide',
                                  'concentration': 'curvepoint'}, inplace=True)

        # remove points that didn't have a mapping (NA)
        df_melted['curvepoint'].replace('', np.nan, inplace=True)
        df_melted.dropna(subset=['curvepoint'], inplace=True)

        df_melted['area'] = df_melted['area'].fillna(0)  # replace NA with 0

    # If dia-nn *.pr_matrix.tsv input file, supply warning about normalizations 
    # and suggest using diann_report.tsv instead
    elif 'Stripped.Sequence' in header_line and 'Precursor.Quantity' not in header_line:
        sys.stdout.write('Input assumed to be DIA-NN *.pr_matrix.tsv filetype.\n')
        sys.stdout.write('WARNING! Use DIA-NN diann_report.tsv instead of pr_matrix!\n')

        df = pd.read_table(filename, sep=None, engine='python')

        # create unique precursor entries so precursors aren't double-counted for curve fitting
        df['Precursor.Charge'] = df['Precursor.Charge'].astype(str)
        df['Modified.Sequence'] = df['Modified.Sequence'].astype(str)
        df['peptide'] = df['Modified.Sequence'] + "_" + df['Precursor.Charge']
        #print(df.head())

        df = df.drop(['Protein.Group',
            'Modified.Sequence',
            'Protein.Ids',
            'Protein.Names',
            'Genes',
            'First.Protein.Description',
            'Proteotypic',
            'Stripped.Sequence',
            'Precursor.Charge',
            'Precursor.Id'], axis=1)  # make a quantitative df with just curve points and peptides
        col_conc_map = pd.read_csv(col_conc_map_file)
        df = df.rename(columns=col_conc_map.set_index('filename')['concentration'])  # map filenames to concentrations

        df_melted = pd.melt(df, id_vars=['peptide'])
        df_melted.columns = ['peptide', 'curvepoint', 'area']
        df_melted = df_melted[df_melted['curvepoint'].isin(col_conc_map['concentration'])]

        # remove colons in Unimod description, e.g. "AAVDC(UniMod:4)EC(UniMod:4)EFQNLEHNEK.png"
        df_melted['peptide'] = df_melted['peptide'].str.replace(':', '')
        #print(df_melted.head())

    # If dia-nn diann_report.tsv input file, check for BOTH Stripped.Sequence AND Precursor.Quantity
    # TODO: add filereading for newer DIA-NN version's diann_report parquet filetype??
    elif 'Stripped.Sequence' in header_line and 'Precursor.Quantity' in header_line:
        sys.stdout.write('Input assumed to be DIA-NN diann_report.tsv filetype.\n')

        df = pd.read_table(filename, sep=None, engine='python')

        # Drop all other columns in the report
        columns_to_keep = ['Precursor.Id', 'File.Name', 'Precursor.Quantity']
        df = df[columns_to_keep]

        # Clean up the column names to match downstream convention
        df.rename(columns={'File.Name': 'filename',
                    'Precursor.Id': 'peptide',
                    'Precursor.Quantity': 'area'}, inplace=True)

        # Map filenames to concentrations - using merge instead of rename
        col_conc_map = pd.read_csv(col_conc_map_file)
        df = pd.merge(df, col_conc_map[['filename', 'concentration']], on='filename', how='inner')

        # Create melted dataframe preserving all values
        df_melted = pd.DataFrame({
            'peptide': df['peptide'],
            'curvepoint': df['concentration'],
            'area': df['area']
        })

        # remove colons in Unimod description, e.g. "AAVDC(UniMod:4)EC(UniMod:4)EFQNLEHNEK.png"
        df_melted['peptide'] = df_melted['peptide'].str.replace(':', '')
        #print(df_melted.head())

    # Spectronaut output
    elif "PEP.StrippedSequence" in header_line:
        sys.stdout.write('Input identified as Spectronaut output.\n')

        df = pd.read_table(filename, sep=None, engine='python')
        
        # create unique precursor entries so precursors aren't double-counted for curve fitting
        df['Precursor.Charge'] = df['EG.PrecursorId'].str.split(".", expand=True)[1]
        df['Modified.Sequence'] = df['EG.PrecursorId'].str.split(".", expand=True)[0].str.strip("_")
        df['peptide'] = df['Modified.Sequence'] + "_" + df['Precursor.Charge']
        
        df.drop(["PG.ProteinGroups",
                 "PG.Organisms",
                 "PG.ProteinNames",
                 "PEP.StrippedSequence",
                 "PEP.PeptidePosition",
                 "EG.PrecursorId",
                 "EG.ModifiedSequence"], axis="columns", inplace=True)  # make a quantitative df with just curve points and peptides
        
        col_conc_map = pd.read_csv(col_conc_map_file)

        df = df.rename(columns=col_conc_map.set_index('filename')['concentration'])  # map filenames to concentrations

        df_melted = pd.melt(df, id_vars=['peptide'])
        df_melted.columns = ['Peptide', 'curvepoint', 'area']
        df_melted = df_melted[df_melted['curvepoint'].isin(col_conc_map['concentration'])]

        # remove colons in Unimod description, e.g. "AAVDC(UniMod:4)EC(UniMod:4)EFQNLEHNEK.png"
        df_melted['Peptide'] = df_melted['Peptide'].str.replace(':', '')

    # convert the curve points to numbers so that they sort correctly
    df_melted['curvepoint'] = pd.to_numeric(df_melted['curvepoint'])

    # replace NaN values with zero
    # TODO: is this appropriate? it's required for lmfit in any case
    # df_melted['area'].fillna(0, inplace=True)  # FutureWarning: this will be deprecated in a future version of pandas
    df_melted.fillna({'area': 0}, inplace=True)

    return df_melted


# associates a multiplier value to the curvepoint a la single-point calibration
def associate_multiplier(df, multiplier_file):
    mutliplier_df = pd.read_csv(multiplier_file)

    # merge the multiplier with the data frame
    merged_df = pd.merge(df, mutliplier_df, on='peptide', how='inner')
    merged_df['curvepoint_multiplied'] = merged_df['curvepoint'] * merged_df['multiplier']
    multiplied_df = merged_df[['peptide', 'curvepoint_multiplied', 'area']]
    multiplied_df.columns = ['peptide', 'curvepoint', 'area']

    return multiplied_df

# makes an educated guess for better "fit_by_lmfit_yang" starting parameters
def linregress(data):
    x = data["curvepoint"]
    y = data["area"]
    w = data["weight"]

    model = LinearModel()

    pars = model.guess(y, x=x)
    result = model.fit(y, pars, x=x, weights=w)

    return result.params["slope"].value, result.params["intercept"].value


# yang's solve for the piecewise fit using lmfit Minimize function
def fit_by_lmfit_yang(x, y, model):

    # residual function
    def fcn2min(params, x, data, weight):
        a = params['a'].value
        b = params['b'].value
        c = params['c'].value
        model_vals = np.maximum(c, a * x + b)
        return (model_vals - data) * weight

    # parameter initialization
    def initialize_params(x, y, weights):
        subsetdf = pd.DataFrame({"curvepoint": pd.to_numeric(x), "area": y, "weight": weights})

        # Initial guess for where noise is
        curvepoints = list(sorted(subsetdf["curvepoint"].unique()))
        noise_mask = subsetdf["curvepoint"].apply(lambda x: x in curvepoints[:2])

        noise_intercept = np.mean(subsetdf["area"][noise_mask])

        # Use linear regression above intersection
        reg_data = subsetdf[~noise_mask]
        linear_slope, linear_intercept = linregress(reg_data)

        # if the noise intercept is lower than the linear intercept, increase it to the linear
        if noise_intercept <= linear_intercept:
            noise_intercept = linear_intercept * 1.05

        return linear_slope, linear_intercept, noise_intercept
    
    def initialize_params_legacy(x, y):
        # 2019 Pino model uses slope from two highest points, intercept at lowest
        subsetdf = pd.DataFrame({'curvepoint': pd.to_numeric(x), 'area': y})
        mean_y = subsetdf.groupby('curvepoint')['area'].mean()  # find the mean response area for each curve point

        # find the top point, second-top point, and bottom points of the curve data
        conc_list = list(set(x))
        top_point = max(conc_list)
        conc_list.remove(top_point)
        second_top = max(conc_list)
        bottom_point = min(conc_list)

        # using the means, calculate a slope (y1-y2/x1-x2)
        linear_slope = (mean_y[second_top]-mean_y[top_point]) / (second_top-top_point)
        # find the noise intercept using average of bottom three points
        noise_intercept = mean_y[bottom_point]
        # find the linear intercept using linear slope (b = y-mx) and the top point
        linear_intercept = mean_y[top_point] - (linear_slope*top_point)

        # edge case catch?
        if noise_intercept < linear_intercept:
            noise_intercept = linear_intercept * 1.05

        return linear_slope, linear_intercept, noise_intercept

    # always compute weights
    weights = np.minimum(1.0 / (np.sqrt(x) + np.finfo(float).eps), 1000)

    # initialize Parameters object
    params = Parameters()

    # build Parameters() differently per user-specified model
    if model == 'piecewise':
        # if the model is piecewise, use the legacy method
        initial_a, initial_b, initial_c = initialize_params_legacy(x,y)
        initial_cminusb = initial_c - initial_b
        params.add('a', value=initial_a, min=0.0, vary=True)  # slope signal
        params.add('b', value=initial_b, vary=True)  # intercept signal
        params.add('c_minus_b', value=initial_cminusb, min=0.0, vary=True)
        params.add('c', expr='b + c_minus_b')

    elif model == "auto":
        # otherwise, use seth's improved initialization method
        initial_a, initial_b, initial_c = initialize_params(x, y, weights)
        params.add('a',         value=initial_a,     min=0.0, vary=True)
        params.add('b',         value=initial_b,             vary=True)
        params.add('c_minus_b', value=(initial_c - initial_b), min=0.0, vary=True)
        params.add('c',         expr='b + c_minus_b')

    # run lmfit
    minner = Minimizer(fcn2min, params, fcn_args=(x, y, weights))
    result = minner.minimize()
    return result, minner


# find the intersection of the noise and linear regime
def calculate_lod(model_params, df, std_mult, min_noise_points, min_linear_points, x, model='auto'):

    """
    Compute LOD (limit of detection) from model_params and bootstraps.
    model='new'      => apply your improved min_point logic
    model='original' => simple intersection+std as in 2021 script
    Returns (LOD, std_noise).
    """
    m_noise, b_noise, m_linear, b_linear = model_params
    
    if model == 'piecewise':

        # calculate the standard deviation for the noise segment
        intersection = (b_linear-b_noise) / (m_noise-m_linear)
        std_noise = np.std(df['area'].loc[(df['curvepoint'].astype(float) < intersection)], ddof=1)  # sample std (Bessel-corrected)

        if m_linear < 0:  # catch edge cases where there is only noise in the curve
            LOD = float('Inf')
        else:
            LOD = (b_noise + (std_mult*std_noise) - b_linear) / m_linear
        lod_results = [LOD, std_noise]

        # LOD edge cases
        curve_points = set(list(df['curvepoint']))
        curve_points.remove(min(curve_points))
        curve_points.remove(max(curve_points))  # now max is 2nd highest point
        if LOD > max(x):  # if the intersection is higher than the top point of the curve or is a negative number,
            lod_results = [float('Inf'), float('Inf')]
        elif LOD < float(min(curve_points)):  # if there's not at least two points below the LOD
            lod_results = [float('Inf'), float('Inf')]

        return LOD, std_noise

    # calculate the standard deviation for the noise segment
    if (m_noise - m_linear) == 0:
        intersection = np.inf
    else:
        intersection = (b_linear-b_noise) / (m_noise-m_linear)

    std_noise = np.std(df['area'].loc[(df['curvepoint'].astype(float) < intersection)], ddof=1)  # sample std (Bessel-corrected)

    min_curvepoint = df["curvepoint"].astype(float).min()
    if intersection <= min_curvepoint and min_noise_points < 1:
        LOD = min_curvepoint ## TODO replace this with whatever the analchem definition of LOD
        std_noise = np.nan
    elif m_linear <= 0:  # catch edge cases where there is only noise in the curve
        LOD = float('Inf')
    else:
        LOD = (b_noise + (std_mult*std_noise) - b_linear) / m_linear
    lod_results = [LOD, std_noise]

    # LOD edge cases
    mask = df["curvepoint"].astype(float) >= LOD
    if df["curvepoint"][mask].nunique() < min_linear_points:
        # if the intersection is higher than the top point of the curve
        lod_results = [np.inf, np.inf]
    elif df["curvepoint"][~mask].nunique() < min_noise_points:
        # if there's not enough below the LOD
        lod_results = [np.inf, np.inf]

    return lod_results


# find the intersection of the noise and linear regime
def calculate_loq(model_params, boot_results, cv_thresh=0.2):

    # initialize the known LOD and a 'blank' LOQ
    LOD = model_params[4]
    LOQ = float('Inf')

    if boot_results.empty:
        LOQ = float('Inf')
    else:
        # subset the bootstrap results for just those values above the LOD
        boot_subset = boot_results[(boot_results['boot_x'] > LOD)]

        # Mask picking out good CVs
        good_cv = boot_subset["boot_cv"] < cv_thresh

        if 0 == good_cv.sum():
            LOQ = float('Inf')
        else:
            # LOQ is the lowest concentration at which CV < threshold
            LOQ = boot_subset[good_cv]['boot_x'].min()

            # LOQ edge cases
            if LOQ >= boot_results['boot_x'].max() or LOQ <= 0:
                LOQ = float('Inf')

    return LOQ


# Module-level function so it can be referenced by bootstrap_many.
# Uses a per-replicate SeedSequence for thread-safe, reproducible resampling.
def _bootstrap_once(df, new_x, seed, model="auto"):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    while True:
        resampled_df = df.sample(n=len(df), replace=True, random_state=rng)
        if resampled_df['area'].nunique() > 1:
            break

    boot_x = np.array(resampled_df['curvepoint'], dtype=float)
    boot_y = np.array(resampled_df['area'], dtype=float)
    fit_result, mini_result = fit_by_lmfit_yang(boot_x, boot_y, model)

    a = fit_result.params['a'].value
    b = fit_result.params['b'].value
    c = fit_result.params['c'].value

    return np.maximum(new_x * a + b, c)


# determine prediction interval by bootstrapping
def bootstrap_many(df, new_x, num_bootreps=100, model="auto"):

    if df.empty or np.isnan(new_x).any():
        boot_summary = pd.DataFrame(columns=['boot_x', 'count', 'mean', 'std', 'min',
                                             '5%', '50%', '95%', 'max', 'boot_cv'])

    else:
        # Bootstrap the data (e.g. resample the data with replacement), eval prediction (new_y) at each new_x
        rows = [_bootstrap_once(df, new_x, i, model) for i in range(num_bootreps)]

        # reshape the bootstrap results: rows=replicates, columns=x-grid points
        mat = np.vstack(rows)   # shape (num_bootreps, len(new_x))

        # calculate lower and upper 95% PI using numpy directly (avoids pandas describe overhead)
        boot_summary = pd.DataFrame({
            'boot_x': new_x,
            'count':  float(mat.shape[0]),
            'mean':   mat.mean(axis=0),
            'std':    mat.std(axis=0, ddof=1),
            'min':    mat.min(axis=0),
            '5%':     np.percentile(mat, 5,  axis=0),
            '50%':    np.percentile(mat, 50, axis=0),
            '95%':    np.percentile(mat, 95, axis=0),
            'max':    mat.max(axis=0),
        })

        # calculate the bootstrapped CV
        boot_summary['boot_cv'] = boot_summary['std']/boot_summary['mean']

    return boot_summary


# plot results
def build_plots(peptide, x, y, model_results, boot_results, num_bootreps, std_mult, cv_thresh, output_dir):

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.figure(figsize=(5, 7))
    plt.suptitle(peptide)

    slope_noise, intercept_noise, slope_linear, intercept_linear, LOD, std_noise, LOQ = model_results

    # plot a line given a slope and intercept
    def add_line_to_plot(slope, intercept, scale, setstyle='-', setcolor='k'):
        axes = plt.gca()
        xlims = np.array(axes.get_xlim())
        x_vals = np.arange(xlims[0], xlims[1], ((xlims[1] - xlims[0]) / 100))
        y_vals = intercept + slope * x_vals
        if scale == 'semilogx':
            plt.semilogx(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        elif scale == 'loglog':
            plt.loglog(x_vals, y_vals, linestyle=setstyle, color=setcolor)
        else:
            plt.plot(x_vals, y_vals, linestyle=setstyle, color=setcolor)

    ###
    ### top plot: linear scale x axis
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'o')  # scatterplot of the data
    if not boot_results.empty:
        plt.fill_between(boot_results['boot_x'],
                         boot_results['mean']-boot_results['std'], boot_results['mean']+boot_results['std'],
                         color='y', alpha=0.3)
    add_line_to_plot(slope_noise, intercept_noise, 'linear', '-', 'g')  # add noise segment line
    add_line_to_plot(slope_noise, (intercept_noise + (std_mult*std_noise)), 'linear', '--', setcolor='0.5')
    if slope_linear > 0:  # add linear segment line
        add_line_to_plot(slope_linear, intercept_linear, 'linear', '-', 'g')

    plt.axvline(x=LOD,
                color='m',
                label=('LOD = %.3e' % LOD))

    plt.axvline(x=LOQ,
                color='c',
                label=('LOQ = %.3e' % LOQ))

    plt.ylabel('signal')

    # force axis ticks to be scientific notation so the plot is prettier
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish
    plt.ylim(ymin=min(y)-max(y)*0.01, ymax=(max(y))*1.05)


    ###
    ### bottom plot: bootstrapped CVs for discretized points
    plt.subplot(2, 1, 2)
    plt.plot(boot_results['boot_x'], boot_results['boot_cv'], marker=None, color='k', label='_nolegend_')

    # plot actual CVs at each curve point
    df = pd.DataFrame({"curvepoint": x, "area": y})

    # Instead of direct CV, match prediction boostrap and take CV of resampled _means_
    # Collect per-rep means into a list, then concat once (avoids O(n²) concat-in-loop)
    boot_means_list = []
    for i in range(num_bootreps):
        resampled_df = df.sample(n=len(df), replace=True)
        boot_means_list.append(resampled_df.groupby("curvepoint")["area"].mean())

    # Compute the average/std across boostrapped means for each point
    means = pd.concat(boot_means_list, axis=1).T
    cv = means.std(axis=0) / means.mean(axis=0)

    plt.scatter(cv.index, cv, marker="o", color="tab:blue", label="_nolegend_")

    plt.axvline(x=LOD,
                color='m',
                label=('LOD = %.3e' % LOD))

    plt.axvline(x=LOQ,
                color='c',
                label=('LOQ = %.3e' % LOQ))

    # add 20%CV reference line
    plt.axhline(y=cv_thresh, color='r', linestyle='dashed')

    #plt.title(peptide, y=1.08)
    plt.xlabel('quantity')
    plt.ylabel('CV')

    # force axis ticks to be scientific notation so the plot is prettier
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.xlim(xmin=min(x)-max(x)*0.01)  # anchor x and y to 0-ish.
    if len(boot_results['boot_cv']) > 0:
        plt.ylim(ymin=-0.01,
                 ymax=(max(boot_results['boot_cv'].max(), cv.max())*1.05))

    # save the figure
    # add legend with LOD and LOQ values
    legend = plt.legend(loc=8, bbox_to_anchor=(0, -.75, 1., .102), ncol=2)
    plt.savefig(os.path.join(output_dir, peptide + '.png'),
                bbox_extra_artists=(legend,),
                bbox_inches='tight', pad_inches=0.75)
    #plt.show()
    plt.close()


def process_peptide(bootreps, cv_thresh, output_dir, peptide, plot_or_not, std_mult, min_noise_points, min_linear_points, subset, verbose, model_choice):
    # sort the dataframe with x values in strictly ascending order
    subset = subset.sort_values(by='curvepoint', ascending=True)

    # create the x and y arrays
    x = np.array(subset['curvepoint'], dtype=float)
    y = np.array(subset['area'], dtype=float)

    # set up the model and the parameters
    result, minner = fit_by_lmfit_yang(x, y, model_choice)
    slope_noise = 0.0  # noise segment is flat
    slope_linear = result.params['a'].value
    intercept_linear = result.params['b'].value
    intercept_noise = result.params['c'].value

    model_parameters = np.asarray([slope_noise, intercept_noise, slope_linear, intercept_linear])

    lod_vals = calculate_lod(model_parameters, subset, std_mult, min_noise_points, min_linear_points, x, model_choice)
    LOD, std_noise = lod_vals

    model_parameters = np.append(model_parameters, lod_vals)

    if not np.isfinite(LOD):
        LOQ = np.inf
        bootstrap_df = bootstrap_many(subset, [np.nan], num_bootreps=0)  # shortcut to get empty DF
    else:
        # calculate coefficients of variation for discrete bins over the linear range (default bins=100)
        x_i = np.linspace(LOD, max(x), num=100, dtype=float)

        bootstrap_df = bootstrap_many(subset, new_x=x_i, num_bootreps=bootreps, model=model_choice)

        if verbose == 'y':
            bootstrap_df.to_csv(path_or_buf=os.path.join(output_dir,
                                                        'bootstrapsummary_' + peptide + '.csv'),
                               index=True)

        LOQ = calculate_loq(model_parameters, bootstrap_df, cv_thresh)

    model_parameters = np.append(model_parameters, LOQ)

    if plot_or_not == 'y':
        # make a plot of the curve points and the fit, in both linear and log space
        # build_plots(x, y, model_parameters, bootstrap_df, std_mult)
        try:
            build_plots(peptide, x, y, model_parameters, bootstrap_df, bootreps, std_mult, cv_thresh, output_dir)
            # continue
        except ValueError:
            sys.stderr.write('ERROR! Issue with peptide %s. \n' % peptide)

    # make a dataframe row with the peptide and its figures of merit
    new_row = [peptide, LOD, LOQ, slope_linear, intercept_linear, intercept_noise, std_noise]
    new_df_row = pd.DataFrame([new_row], columns=['peptide', 'LOD', 'LOQ',
                                                  'slope_linear', 'intercept_linear', 'intercept_noise',
                                                  'stndev_noise'])

    return new_df_row


def main():
    _default_threads = max(1, (os.cpu_count() or 1) - 2)
    # usage statement and input descriptions
    parser = argparse.ArgumentParser(
        description='A  model for fitting calibration curve data. Takes calibration curve measurements as input, and \
                    returns the Limit of Detection (LOD) and Limit of Quantitation (LOQ) for each peptide measured in \
                    the calibration curve.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('curve_data', type=str,
                        help='a matrix containing peptides and their quantitative values across each curve point (currently\
                                supporting Encyclopedia *.elib.peptides.txt quant reports and Skyline export reports)')
    parser.add_argument('filename_concentration_map', type=str,
                        help='a comma-delimited file containing maps between filenames and the concentration point \
                                they represent (two columns named "filename" and "concentration")')
    parser.add_argument('--std_mult', default=2, type=float,
                        help='specify a multiplier of the standard deviation of the noise for determining limit of \
                        detection (LOD)')
    parser.add_argument('--cv_thresh', default=0.2, type=float,
                        help='specify a coefficient of variation threshold for determining limit of quantitation (LOQ) \
                                (Note: this should be a decimal, not a percentage, e.g. 20%% CV threshold should be input as \
                                0.2)')
    parser.add_argument('--bootreps', default=100, type=int,
                        help='specify a number of times to bootstrap the data (Note: this must be an integer, e.g. to \
                                resample the data 100 times, the parameter value should be input as 100')
    parser.add_argument('--min_noise_points', default=DEFAULT_MIN_NOISE_POINTS, type=int,
                        help="specify the minimum required curve points required below the LOD")
    parser.add_argument('--min_linear_points', default=DEFAULT_MIN_LINEAR_POINTS, type=int,
                        help="specify the minimum required curve points required above the LOD")
    parser.add_argument('--multiplier_file', type=str,
                        help='use a single-point multiplier associated with the curve data peptides')
    parser.add_argument('--output_path', default=os.getcwd(), type=str,
                        help='specify an output path for figures of merit and plots')
    parser.add_argument('--plot', default='y', type=str,
                        help='yes/no (y/n) to create individual calibration curve plots for each peptide')
    parser.add_argument('--verbose', default='n', type=str,
                        help='output a detailed summary of the bootstrapping step')
    parser.add_argument('--model', default='auto', choices=['auto', 'piecewise'], 
                        help='Specify which model to use for LOQ fitting, auto (per peptide) or the original piecewise fit. \
                                Default is auto (best AIC).')
    parser.add_argument('--n_threads', default=_default_threads, type=int,
                        help='number of worker processes for parallel peptide processing '
                             f'(default: cpu_count - 2 = {_default_threads}; set to -1 to use all CPUs)')


    # parse arguments from command line
    args = parser.parse_args()
    raw_file = args.curve_data
    col_conc_map_file = args.filename_concentration_map
    cv_thresh = args.cv_thresh
    std_mult = args.std_mult
    bootreps = args.bootreps
    min_noise_points = args.min_noise_points
    min_linear_points = args.min_linear_points
    multiplier_file = args.multiplier_file
    output_dir = args.output_path
    plot_or_not = args.plot
    verbose = args.verbose
    model_type = args.model
    n_threads = args.n_threads if args.n_threads != -1 else None  # None → ProcessPoolExecutor uses all CPUs

    # read in the data
    quant_df_melted = read_input(raw_file, col_conc_map_file)

    # associate multiplier with the curvepoint ratio (if there is a multiplier provided)
    if multiplier_file:
        quant_df_melted = associate_multiplier(quant_df_melted, multiplier_file)

    # initialize empty data frame to store figures of merit
    peptide_fom = pd.DataFrame(columns=['peptide', 'LOD', 'LOQ',
                                        'slope_linear', 'intercept_linear', 'intercept_noise',
                                        'stndev_noise'])

    # and awwaayyyyy we go~
    with ProcessPoolExecutor(max_workers=n_threads) as exec:
        # First, submit each peptide as a job to the executor
        futures = []
        for peptide, subset in quant_df_melted.groupby('peptide'):
            if subset.empty:  # if the peptide is nan, skip it and move on to the next peptide
                continue

            futures.append(exec.submit(process_peptide, bootreps, cv_thresh, output_dir, peptide, plot_or_not, std_mult, min_noise_points, min_linear_points, subset, verbose, model_type))

        # Create output file with headers
        output_file = os.path.join(output_dir, 'figuresofmerit.csv')
        headers = ['peptide', 'LOD', 'LOQ', 'slope_linear', 'intercept_linear', 'intercept_noise', 'stndev_noise']
        with open(output_file, 'w') as f:
            f.write(','.join(headers) + '\n')
            f.flush()
            os.fsync(f.fileno())

        # Process results as they complete so that if it randomly errors out, you at least have some output results
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result_df = future.result()
                # Use a file lock to prevent concurrent writes from the threads
                with open(output_file, 'a') as f:
                    result_df.to_csv(f, header=False, index=False)
                    f.flush()
                    os.fsync(f.fileno())
                peptide_fom = pd.concat([peptide_fom, result_df], ignore_index=True, axis=0)
            except Exception as e:
                print(f"Error processing result: {str(e)}")

    #peptide_fom.to_csv(path_or_buf=os.path.join(output_dir, 'figuresofmerit.csv'),
    #                   index=False)


if __name__ == "__main__":
    main()